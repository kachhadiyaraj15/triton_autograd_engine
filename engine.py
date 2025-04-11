"""
This document and the object TritonTensor are the foundation of our autograd engine that everything else runs off of.
Triton only works on pytorch's torch.tensor object so what we're doing here is building a wrapper function TritonTensor around torch.tensor that allows us to only use our own custom kernel any of pytorch's.

in this document we'll use"torch.tensor" to refer to pytorch's implementation of tensors,
"TritonTensor" to refer to our implementation which is itself a wrapper around pytorch's, and "tensor" to refer to them both at once
"""

from typing import Union, Tuple, Optional
import numpy as np
from math import prod

import torch
import triton
import triton.language as tl
from kernels import elementwise, matmul, vectorwise

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")
properties = triton.runtime.driver.active.utils.get.get_device_properties(DEVICE.index)
TOTAL_SRAM_PER_SM = properties["max_shaed_mem"] # each SM has a fixed amount of SRAM that it can access 
# if  one SM isn't using all its available SRAM then another can be spun up to use the remainder


class TritonTensor:
    '''
    Stores a tensor ans its gradient information.
    Functions as a wrapper aound torch.tensor so that we can take advantageof in-built ops
    like __add__, __mul__ rather than writing Triton kernels in the tranditional way.
    '''
    def __init__(self, 
                 data : Union[float, int, list, np.ndarray, torch.Tensor],
                 requires_grad : bool = False,
                 device : Optional[Union[str, torch.device]] = None,
                 _children : Tuple["Tensor", ...] = ()):
        
        # convert input data to torch.Tensor if it isn't already
        if isinstance(data, torch.Tensor):
            self.data = data.to(torch.float32)
                # we're enforcingor autograd engine only use fp32 for simplicity's sake
        else:
            self.data = torch.tensor(data, dtype = torch.float32, requires_grad = False)
                # requires_grad = False prevents pytorch from taking up memory by keeping track of its own gradient 
        
        # Move tensor to specified device (default to CUDA if available)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = self.data.to(device)
        
        # store tensor metadata 
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.dtype = self.data.dtype
        self.device = self.data.device
        self.numel = lambda: self.data.numel()
        
        # gradient-related attributes
        self.requires_grad = requires_grad
        self.grad = torch.zeros_like(self.data) if requires_grad else None
        
        # Autograd graph information
        self._prev = set(_children)
        self._backward = lambda: None # function to compute local gradient updates
    
    def __repr__(self):
        return f"TritonTensor:\n{self.data}" # \nGrad:{self.grad}
    
    def _binary(self, other, op):
        '''
        a simple elementwise binary operation that supports broadcasting of "other" up to size of self
        '''
        assert self.device == other.device, \
            f"tensors must be on the same device but got self.device: {self.device}, other.device: {other.device}"
        assert self.data.is_contiguous() and other.data.is_contiguous()
        
        # convert other to Tensor if needed
        other = other if isinstance(other, TritonTensor) else TritonTensor(other)
        
        # getting the total number of entries of each of our inputs 
        n_elements = self.numel()
        loop_stride= other.numel() # the name `loop_strides` will make sense in the kernel
        
        # forcing other to be the smaller tensor that needs to get broadcasted into self makes our logic simpler
        assert n_elements >= loop_stride, "for addition, the first input mush have more than or as many entries of the second"
        assert n_elements % loop_stride == 0, "the number of entries in the first input must be a multiple of the second"
        
        # restricting the possible set of inputs to those which are logically broadcastable
        # if we didn't do this then later our kernel would compute nonsensical broadcasting values
        if self.shape != other.shape and other.shape != (1,): # the latter case is for when other is a single scalar 
            ptr = 0
            for d in self.shape:
                if ptr == other.ndim: break
                if d == other.shape[ptr]:
                    ptr += 1
            assert ptr == other.ndim, \
                f"for broadcasing to work, all dims in a ({self.shape}) must be a subset of those in b ({other.shape})"
        # TODO is this restriction good enough? am i missing something?
        
        # Preallocating the output 
        output = torch.empty_like(self.data)
        
        # Define grid based on tensor dimensions
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        # Launch kernel
        elementwise.binary_op_forwad[grid](
            self.data, other.data, output,
            n_elements, loop_stride,
            OP = op, # designates which operation to run (addition, subtraction, multiplication, division)
        )
        
        # wrap output in TritonTensor with autograd information
        out = TritonTensor(output, requires_grad = (self.requires_grad or other.requires_grad), _children = (self, other))
        
        # define our backward pass
        def _backward():
            if self.requires_grad:
                elementwise.binary_op_backward_dx[grid](
                    other.data, 
                    self.grad,
                    out.grad,
                    n_elements, loop_stride,
                    OP = op,
                )
            if other.requires_grad:
                elementwise.binary_op_backward_dy[grid](
                    self.data, 
                    other.grad,
                    out.grad,
                    n_elements, loop_stride,
                    OP = op,
                )
        out._backward = _backward
        return out
    
    def __add__(self, other):
        return self._binary(other, op = "add")
    
    def __mul__(self, other):
        return self._binary(other, op = "mul")
    
    def __sub__(self, other):
        return self._binary(other, op = "sub")
    
    def __truediv__(self, other):
        """Placeholder for negation"""
        # TODO : Implement Triton kernel for negation
        raise NotImplementedError("Negation kernel not implemented yet.")
    
    def __matmul__(self, other):
        """
        matmul implementation built to support only the shape we need, so
        A: tensor @ B : tensor = C:tensor for the self-attention mechanism and 
        A: tensor @ B : tensor = C:tensor for linear layers
        also we don't need this regular matmul shape layout, but hey why not
        A: matrix @ B:matrix = c:matrix
        """
        # Check constraints 
        assert self.ndim >= 2 and other.ndim >= 2,\
            f"matmul inputs nust be tensors or matrics, not vectors, but got A.ndim={self.ndim}, B.ndim = {other.ndim}"
        assert self.ndim >= other.ndim, \
            f"matmul only supports broadcasting second input tensor, meaning B must have equal to or fewer dims than A"
        assert self.shape[-1] == other.shape[-2], \
            f"incompatible dimensions for matmul, A: {self.shape} and B:{other.shape}"
        if other.ndim > 2:
            assert self.shape[:-2] == other.shape[:-2],\
                f"matmul only supports inputs of same leading dimensions shape, but got A: {self.shape} and B: {other.shape}"
        assert self.data.is_contiguous(), "Matrix A must be contiguous" # TODO but why not other/B as well tho?
        
        # get matrix dimension lengths
        (m,k), n = self.shape[-2:], other.shape[-1]
        # how many **batches** and **heads** to parallelize along, (parallel matrix count)
        parallel_matrix_ct = prod(self.shape[:-2] if self.ndim > 2 else 1)   # TODO - Why?
        
        # allocate output
        #  ------------- ???????????????? -------------------
        out = torch.empty(self.shape[:-2] + (m, n), device = self.device, dtype = torch.float32)  # expmaple: (batch_size, num_heads, m, n)
        
        # 2D launch kernel where each parallel_matrix_ct and each block gets its own program
        grid = lambda meta:(
            triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n , meta["BLOCK_SIZE_N"]),
            parallel_matrix_ct
        )
        
        matmul.matmul_fwd[grid](
            self.data, other.data, out,
            m, n, k,
            self.data.stride(-3) if self.ndim > 2 else 0, self.data.stride(-2), self.data.stride(-1),
            other.data.stride(-3) if other.ndim > 2 else 0, other.data.stride(-2), other.data.stride(-1),
            out.stride(-3) if out.ndim > 2 else 0, out.stride(-2), out.stride(-1),
        )
            # through this kernel we're mutlipling the tensor by tensor matmul, which only gets used in the attention mechanism
            # and tensor by matrix matmul, which gets used for all linear layers
        
        # Wrap output in Tensor with autograd information
        out = TritonTensor(
            out.to(self.data.dtype),
            requires_grad= (self.requires_grad or other.requires_grad),
            _children = (self, other)
        )
        
        # define our backward pass
        def _backward():
            # only run backward kernel if at least one input requires grad
            if self.requires_grad:
                bwd_grid_dA = lambda meta: (
                    triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(k, meta["BLOCK_SIZE_K"]),
                    parallel_matrix_ct,
                )
                
                matmul.matmul_bwd_dA[bwd_grid_dA](
                    other.data, self.grad, out.grad,
                    m, n, k,
                    other.data.stride(-3) if other.ndim > 2 else 0, other.data.stride(-2), other.data.stride(-1),
                    self.grad.stride(-3) if self.ndim > 2 else 0, self.grad.stride(-2), self.grad.stride(-1),
                    out.grad.stride(-3) if out.ndim > 2 else 0, out.grad.stride(-2), out.grad.stride(-1),
                )
            
            if other.requires_grad:
                bwd_grid_dB = lambda meta: (
                    triton.cdiv(k, meta["BLOCK_SIZE_K"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
                    parallel_matrix_ct,
                )
                
                matmul.matmul_bwd_dB[bwd_grid_dB](
                    self.data, other.grad, out.grad,
                    m, n, k,
                    self.data.stride(-3) if self.ndim > 2 else 0, self.data.stride(-2), self.data.stride(-1),
                    other.grad.stride(-3) if other.ndim > 2 else 0, other.grad.stride(-2), other.grad.stride(-1),
                    out.grad.stride(-3) if out.ndim > 2 else 0, out.grad.stride(-2), out.grad.stride(-1),
                )
        out._backward = _backward
        return out
    
    def _unary(self, op):
        assert self.data.is_contiguous(), "Tensor must be contiguous"
        
        # Preallocating the output
        output = torch.empty_like(self.data)
        
        # Define grid based on tensor dimensions
        n_elements = self.data.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        # Launch kernel
        elementwise.unary_op_forward[grid](
            self.data, output,
            n_elements,
            OP = op, # designates which operation to run (addition, subtraction, multiplication, division)
        )
        
        # Wrap output in Tensor with autograd information
        out = TritonTensor(
            output,
            requires_grad = self.requires_grad,
            _children = (self,)
        )
        
        # Define backward pass
        def _backward():
            if self.requires_grad:
                elementwise.unary_op_backward[grid](
                    self.data, self.grad, 
                    out.data, out.grad,
                    n_elements,
                    op,
                )
        out._backward = _backward
        return out
    
    def exp(self):
        return self._unary(op = "exp")
    
    def log(self):
        return self._unary(op = "log")
    
    def relu(self):
        return self._unary(op = "relu")
    
    def _reduction(self, op):
        """
        all reduction ops (sum, min, max, etc) move through here
        it's a relatively inflexible module; we only support reduction along the tensor's final dimension
        """
        assert self.data.is_contiguous(), "Tensor must be contiguous"
        
        # Preallocating the output
        output = torch.empty(self.data.shape[:-1], device = self.device, dtype = self.dtype, requires_grad = False)
        
        # Get tensor dimensions to ensure or parallelization will work
        n_cols = self.shape[-1]
        BLOCK_SIZE_N = triton.next_power_of_2(n_cols)
        # 4 for the 4 bytes in fp32
        assert BLOCK_SIZE_N * 4 < TOTAL_SRAM_PER_SM, \
            f"Vectors (each size {BLOCK_SIZE_N * 4} too large for SRAM, max size {TOTAL_SRAM_PER_SM})"
        
        # Parallelize with multiple rows in a PID
        grid = lambda meta: (
            triton.cdiv(self.data.numel() // n_cols, meta['BLOCK_SIZE_M']),
        )
        # Launch kernel
        vectorwise.reduction_op_forward[grid](
            self.data, output,
            self.data.numel(), output.numel(),
            self.data.stride()[-2], n_cols,
            op, 
            BLOCK_SIZE_N = BLOCK_SIZE_N,
        )
        
        # Wrap output in Tensor with autograd information
        out = TritonTensor(
            output,
            requires_grad = self.requires_grad,
            _children = (self,)
        )
        
        # define our backward pass
        def _backward():
            if self.requires_grad:
                vectorwise.reduction_op_backward[grid](
                    self.data,
                    self.grad, out.grad,
                    self.grad.numel(), out.grad.numel(),
                    self.data.stride()[-2], n_cols, 
                    op, 
                    BLOCK_SIZE_N=BLOCK_SIZE_N,
                )
        out._backward = _backward

        return out
    
    def sum(self):
        return self._reduction(op='sum')

    def mean(self):
        return self._reduction(op='mean')

    def max(self):
        return self._reduction(op='max')

    def min(self):
        return self._reduction(op='min')

    def var(self):
        return self._reduction(op='var')

    def std(self):
        return self._reduction(op='std')

    def softmax(self):
        pass
    
    

        