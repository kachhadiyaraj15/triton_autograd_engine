from typing import Union, Tuple, Optional
import numpy as np
from math import prod, sqrt

import torch
import triton
import triton.language as tl

from engine import TritonTensor

from kernels import elementwise, matmul, vectorwise, modules, flash_attetntion

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")
properties = triton.runtime.driver.active.utils.get.get_device_properties(DEVICE.index)
TOTAL_SRAM_PER_SM = properties["max_shaed_mem"] # each SM has a fixed amount of SRAM that it can access
# if  one SM isn't using all its available SRAM then another can be spun up to use the remainder

BATCH = 32

####################################################################################
###################### Unary Ops ###################################################
####################################################################################

class _unary_op(torch.autograd.Function):
    """a simple unary operation"""
    
    @staticmethod
    def forward(ctx, a, op_name):
        assert a.is_contiguous(), "Input tensor must be contiguous"
        n_elements = a.numel()
        
        # Preallocating the output
        b = torch.empty_like(a)
        
        # Define grid based on tensor dimensions
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        # Launch Kernel
        elementwise.unary_op_forward[grid](
            a, b,
            n_elements,
            op_name, # designates which operation to run (exp, log, relu, etc.)
        )
        
        ctx.save_for_backward(a, b)
        ctx.grid = grid
        ctx.n_elements = n_elements
        ctx.op_name = op_name
        return b
    
    @staticmethod
    def backward(ctx, db):
        a, b = ctx.saved_tensors
        n_elements = ctx.n_elements
        op_name = ctx.op_name
        
        # Preallocating the output gradient
        da = torch.empty_like(a)
        
        # Define grid based on tensor dimensions
        grid = ctx.grid
        
        # Launch Kernel
        elementwise.unary_op_backward[grid](
            a, da,
            b, db,
            n_elements,
            op_name, # designates which operation to run (exp, log, relu, etc.)
        )
        
        return da, None 

unary_op_fn = _unary_op.apply

def get_unary_ops_args(args):
    ops = []
    if args.all or args.exp:
        ops.append("exp")
    if args.all or args.log:
        ops.append("log")
    if args.all or args.relu:
        ops.append("relu")
    return ops

# First define an empty list that will be populated before the decorator is used
unary_op_configs = []
def generate_unary_op_configs(ops):
    configs = []
    for op in ops:
        for mode in ["fwd", "bwd"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names = ["tot_elements"],
                    x_vals = [2**i for i in range(12,16,1)],
                    line_args = "provider",
                    line_vals = ["torch", "triton"],
                    line_names = ["PyTorch", "Triton"],
                    styles = [("blue", "-"), ("red", "-")],
                    ylabel = "GB/s",
                    x_label = "Total elements per output tensor",
                    plot_name = f"{op}_{mode}",
                    args = {"op": op, "mode": mode,},
                )
            )
    return configs

@triton.testing.perf_report(unary_op_configs)
def benchmark_unary(tot_elements, provider, op, mode, device = DEVICE):
    """

    Benchmark Triton kernels for unary operations.
    Args :
        tot_elements : Total number of elements in the input tensor.
        provider : Provider of the implementation (PyTorch or Triton).
        op : Operation to benchmark (exp, log, relu).
        mode : Mode to benchmark (fwd or bwd).
    device : Device to run the benchmark on (default: current CUDA device).
    """
    # Generte input data 
    dim = int(tot_elements ** 0.5)
    A = torch.randn((BATCH, dim, dim), device = device, dtype = torch.float32, requires_grad = True)
    
    # Select implementation
    if op == "exp":
        fn = lambda: unary_op_fn(A, op) if provider == "triton" else torch.exp(A)
    if op == "log":
        fn = lambda: unary_op_fn(A, op) if provider == "triton" else torch.log(A)
    if op == "relu":
        fn = lambda: unary_op_fn(A, op) if provider == "triton" else torch.relu(A)
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda : O.backward(dO, retain_grpah = True)

####################################################################################
###################### Binary Ops ###################################################
####################################################################################       

class _binary_op(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, a, b, op_name):
        """
        a simple element-wise binary operation that supports broadcasting of "b" up to tot_elements `a`
        """
        assert a.device == b.device
        assert a.is_contiguous() and b.is_contiguous()
        
        # getting the total number of entites of each of our inputs
        n_elements = a.numel()
        loop_stride = b.numel() # the name `loop_strides` will make sense in the kernel
        
        # restricting the possible set of inputs to those which are logically broadcastable
        # if we didn't do this then later our kernel would compute nonsensical broadcasting values
        if a.shape != b.shape:
            ptr = 0
            for d in a.shape:
                if ptr == b.ndim: break
                if d == b.shape[ptr]:
                    ptr += 1
            assert ptr == b.ndim, \
                f"for broadcasing to work, all dims in a ({a.shape}) must be a subset of those in b ({b.shape})"

        # Preallocating the output
        c =torch.empty_like(a)
        
        # Define grid based on tensor dimensions
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        # Launch Kernel
        elementwise.binary_op_forward[grid](
            a, b, c, 
            n_elements, loop_stride,
            op_name, # designates which operation to run (addition, subtraction, multiplication, division)
        )
        
        ctx.save_for_backward(a, b)
        ctx.grid = grid
        ctx.n_elements = n_elements
        ctx.op_name = op_name
        ctx.loop_stride = loop_stride
        return c

    @staticmethod
    def backward(ctx, dc):
        a,b = ctx.saved_tensors
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        
        # reusing the grid from the forward pass
        elementwise.binary_op_backward_dx[ctx.grid](
            b, da, dc,
            ctx.n_elements, ctx.loop_stride,
            ctx.op_name, # designates which operation to run (addition, subtraction, multiplication, division)
        )
        
        return da, db, None

binary_op_fn = _binary_op.apply

def get_binary_ops_args(args):
    ops = []
    if args.all or args.add:
        ops.append("add")
    if args.all or args.sub:
        ops.append("sub")
    if args.all or args.mul:
        ops.append("mul")
    if args.all or args.div:
        ops.append("div")
    return ops

# First define an empty lit that will be populated before the decoratore is used 
binary_op_configs = []
def generate_binary_op_configs(ops):
    configs = []
    for op in ops:
        for mode in ["fwd", "bwd"]:
            for broadcasting in [True, False]:
                configs.append(
                    triton.testing.Benchmark(
                        x_names = ["tot_elements"],
                        x_vals = [2**i for i in range(12,16,1)],
                        line_args = "provider",
                        line_vals = ["torch", "triton"],
                        line_names = ["PyTorch", "Triton"],
                        styles = [("blue", "-"), ("red", "-")],
                        ylabel = "GB/s",
                        x_label = "Total elements per output tensor",
                        plot_name = f"{op}_{mode}_broadcasting={broadcasting}",
                        args = {"op": op, "mode": mode, "broadcasting": broadcasting},
                    )
                )
    return configs

@triton.testing.perf_report(binary_op_configs)
def benchmark_binary(tot_elements, provider, op, mode, broadcasting, device=DEVICE):
    """
    Benchmark Triton binary operations against PyTorch.
    
    Args:
        tot_elements: Total number of elements in the tensors
        provider: 'torch' or 'triton'
        op: "add", "sub", "mul", or "div"; designates the operation to be performed
        mode: "fwd" or "bwd"
        broadcasting: True for same-size inputs and False for smaller B to be broadcasted
        device: Device to run on
    """
    # Generate input data 
    dim = int(tot_elements ** 0.5)
    A = torch.randn((BATCH, dim, dim), device = device, dtype = torch.float32, requires_grad = True)
    B = torch.randn((dim,) if broadcasting else (BATCH, dim, dim), device = device, dtype = torch.float32, requires_grad = True)
    
    # select implementation
    if op == "add":
        fn = lambda: binary_op_fn(A, B, op) if provider == "triton" else A + B
    if op == "sub":
        fn = lambda: binary_op_fn(A, B, op) if provider == "triton" else A - B
    if op == "mul":
        fn = lambda: binary_op_fn(A, B, op) if provider == "triton" else A * B
    if op == "div":
        fn = lambda: binary_op_fn(A, B, op) if provider == "triton" else A / B
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph = True)
                    
########################################################################################
########################### Matrix Multiplication ############################################
########################################################################################

class _matmul(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, a, b):
        """
        matmul implementation built to support only the shapes we need,so
        A: tensor @ B: tensor = C: tensor for the self-attention mechanism and
        A: tensor @ B: matrix  = C: tensor for linear layers
        also we dont need this regular matmul shape layout, but hey why not 
        A: matrix @ B: matrix = c: matrix
        """
        # Check constraints
        assert a.ndim >= 2 and b.ndim >=2
        assert a.ndim >= b.ndim
        assert a.shape[-2] == b.shape[-1]  
        if b.ndim > 2:
            assert a.shape[:-2] == b.shape[:-2]
        assert a.is_contiguous()
        
        # getting the total number of entries of each of our inputs
        n_elements = a.numel()
        loop_stride = b.numel()
        
        # get matrix dimensions lengths
        (m, k), n = a.shape[-2:], b.shape[-1]
        # how many batches and heads to parallelize along 
        parallel_matrix_ct = prod(a.shape[:-2] if a.ndim > 2 else 1)
        
        # allocates output 
        c = torch.empty(a.shape[:-2] + (m, n), device = a.device, dtype = torch.float32)
        
        # 2D launch kernel where each preceeding_dim and each block gets its own program
        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_SIZE_M"]) * triton.cdiv(n, meta["BLOCK_SIZE_N"]),
            parallel_matrix_ct,
        )
        # Launch kernel
        matmul.matmul_fwd[grid](
            a,b, c,
            m, n, k,
            a.stride(-3) if a.ndim > 2 else 0, a.stride(-2), a.stride(-1),
            b.stride(-3) if b.ndim > 2 else 0, b.stride(-2), b.stride(-1),
            c.stride(-3) if c.ndim > 2 else 0, c.stride(-2), c.stride(-1),
        )
        
        ctx.save_for_backward(a, b)
        ctx.m = m
        ctx.n = n
        ctx.k = k
        ctx.parallel_matrix_ct = parallel_matrix_ct
        return c
    
    @staticmethod
    def backward(ctx, dc):
        a,b = ctx.saved_tensors
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        
        bwd_grid_dA = lambda meta:[
            triton.cdiv(ctx.m, meta["BLOCK_SIZE_M"]) * triton.cdiv(ctx.k, meta["BLOCK_SIZE_K"]),
            ctx.parallel_matrix_ct,
        ]
        
        matmul.matmul_bwd_dA[bwd_grid_dA](
            b, da, db,
            ctx.m, ctx.n, ctx.k,
            b.stride(-3) if b.ndim > 2 else 0, b.stride(-2), b.stride(-1),
            da.stride(-3) if da.ndim > 2 else 0, da.stride(-2), da.stride(-1),
            db.stride(-3) if db.ndim > 2 else 0, db.stride(-2), db.stride(-1),
        )
        
        bwd_grid_dB = lambda meta:[
            triton.cdiv(ctx.k , meta["BLOCK_SIZE_K"]) * triton.cdiv(ctx.n, meta["BLOCK_SIZE_N"]),
            ctx.parallel_matrix_ct,
        ]
        
        matmul.matmul_bwd_dB[bwd_grid_dB](
            a, db, dc,
            ctx.m, ctx.n, ctx.k,
            a.stride(-3) if a.ndim > 2 else 0, a.stride(-2), a.stride(-1),
            db.stride(-3) if db.ndim > 2 else 0, db.stride(-2), db.stride(-1),
            dc.stride(-3) if dc.ndim > 2 else 0, dc.stride(-2), dc.stride(-1),
        )
        
        return da, db

matmul_fn = _matmul.apply

# Create matmul configs
matmul_configs = []
for mode in ["fwd", "bwd"]:
    for broadcasting in [True, False]:
        matmul_configs.append(
            triton.testing.Benchmark(
                x_names = ["M", "N", "K"], # Argument names to vary
                x_vals = [
                    128 * i for i in range(1, 5, 1)], # Different inpuy tot_elements
                line_args = "provider", # Argument names whose value corresponds to a different line in the plot
                line_vals = ["torch", "triton"], # Values for the line_args
                line_names = ["PyTorch", "Triton"], # Names for the lines in the plot
                styles = [("blue", "-"), ("red", "-")], # Colors and styles for the lines
                ylabel = "TFLOPS", # Y-axis label
                x_label = "M, N and K", # X-axis label
                plot_name = f"matmul_{mode}_broadcasting={broadcasting}", # Name of the plot
                args = {"mode" : mode, "broadcasting" : broadcasting}, # Additional arguments for the benchmark
            )
        )
@triton.testing.perf_report(matmul_configs)
def benchmark_matmul(M, N, K, provider, mode, broadcasting, device = DEVICE):
    A = torch.randn((BATCH, M, K), device = device, dtype = torch.float32, requires_grad = True)
    B = torch.randn((K, N) if broadcasting else (BATCH, K, N), device = device, dtype = torch.float32, requires_grad = True)
    
    if provider == "torch":
        fn = lambda: A @ B
    else:  # triton
        fn = lambda: matmul_fn(A, B)
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph = True)
        
    # for matmul we'll measure TFLOPs instead of GB/s since slops are the limimting factor
    ms = triton.testing.do_benchmark(fn)
    perf = (2 if mode == "fwd" else 4) * BATCH * M * N * K * 1e-12 / ( ms * 1e-3)
    # 2 or 4 = number of operations per entry ( mul and add for fwd & another set for two gradients during bwd)
    # BATCH * M * N * K = number of elements 
    # 1e-12 convert flops to Teraflops
    # ms * 1e-3 convert milliseconds to seconds
    return perf

########################################################################################
########################### Reduction Operations ############################################
########################################################################################

class _reduction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, op_name):
        """
        implementaion of reduction ops
        only suppports reduction along the final dimension
        """
        # check constrains
        assert x.is_contiguous(), "Input tensor must be contiguous"
        
        # allocates output
        y = torch.empty(x.shape[:-1], device = x.device, dtype = torch.float32)
        
        # get tensor dimensions to ensure our parallelization scheme will work
        n_cols = x.shape[-1]
        BLOCK_SIZE_N = triton.next_power_of_2(n_cols)
        # 4 for the 4 bytes in fp32
        assert BLOCK_SIZE_N * 4 < TOTAL_SRAM_PER_SM, \
            f"Vector (each size {BLOCK_SIZE_N * 4}) too large to fir into SRAM size {TOTAL_SRAM_PER_SM} "
        
        # we'll parallelize with multiple rows in a PID
        grid = lambda meta : ( triton.cdiv(x.numel() // n_cols, meta["BLOCK_SIZE_M"]),)
        # Launch Kernel
        vectorwise.reduction_op_forward[grid](
            x, y, 
            x.numel(), y.numel(), 
            x.stride()[-2], n_cols, 
            op = op_name, 
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )

        ctx.save_for_backward(x)
        ctx.op_name = op_name
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.n_cols = n_cols
        ctx.grid = grid
        return y

    @staticmethod
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        dx = torch.empty(x.shape, device=dy.device, dtype=dy.dtype)
        grid = ctx.grid
        
        vectorwise.reduction_op_backward[grid](
            x,
            dx, dy,
            dx.numel(), dy.numel(),
            dx.stride()[-2], ctx.n_cols, 
            op = ctx.op_name, 
            BLOCK_SIZE_N = ctx.BLOCK_SIZE_N,
        )

        return dx, None

reduction_fn = _reduction.apply

# Define the operations list based on input args
def get_reduction_args(args):
    ops = []
    if args.all or args.sum:
        ops.append("sum")
    if args.all or args.mean:
        ops.append("mean")
    if args.all or args.var:
        ops.append("var")
    if args.all or args.std:
        ops.append("std")
    return ops

# First define an empty list that will be populated before the decorator is used
reduction_configs = []
def generate_reduction_configs(ops):
    configs = []
    for op in ops:
        for mode in ["fwd", "bwd"]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=['tot_elements'],
                    x_vals=[2**i for i in range(12, 24, 1)],
                    line_arg='provider',
                    line_vals=['torch', 'triton'],
                    line_names=['PyTorch', 'Triton'],
                    styles=[('blue', '-'), ('red', '-')],
                    ylabel='GB/s',
                    xlabel="Total elements per output tensor",
                    plot_name=f'{op}_{mode}',
                    args={"op": op, "mode": mode,},
                ))
    return configs

@triton.testing.perf_report(reduction_configs)
def benchmark_reduction(tot_elements, provider, op, mode, device=DEVICE):
    """
    Benchmark Triton reduction operations against PyTorch.
    
    Args:
        tot_elements: Total number of elements in the tensors
        provider: 'torch' or 'triton'
        op: "sum", "mean", "max", etc; designates the operation to be performed
        mode: "fwd" or "bwd"
        device: Device to run on
    """
    # Generate input data
    dim = int(tot_elements ** 0.5)
    X = torch.randn((dim, dim), device=device, requires_grad=True)
    
    # Select implementation
    if op == "sum":
        fn = lambda: reduction_fn(X, op) if provider == 'triton' else torch.sum(X, dim=1)
    elif op == "mean":
        fn = lambda: reduction_fn(X, op) if provider == 'triton' else torch.mean(X, dim=1)
    elif op == "var":
        fn = lambda: reduction_fn(X, op) if provider == 'triton' else torch.var(X, dim=1)
    elif op == "std":
        fn = lambda: reduction_fn(X, op) if provider == 'triton' else torch.std(X, dim=1)
    if mode == "bwd":
        O = fn()
        dO = torch.randn_like(O)
        fn = lambda: O.backward(dO, retain_graph=True) 

    # Benchmark
    # for entry-wise operations we'll measure memory throughput since that's the limiting factor
    if mode == "fwd": # all fwd passes have same mem read/write behavior
        gb = 2 * tot_elements * 4 * 1e-9
        # 2 = number of memory operations (1 read + 1 write)
        # 4) = bytes per element (for float32)
        # 1e-9 converts bytes to GB
    elif mode == "bwd" and op in ("sum", "mean"):
        gb = 2 * tot_elements * 4 * 1e-9
    elif mode == "bwd" and op in ("var", "std"):
        gb = 3 * tot_elements * 4 * 1e-9
        # TODO should these two use TFLOPs instead of GB/s? not worth putting in the effort to change tbh
    # 1e-3 converts milliseconds to seconds
    ms = triton.testing.do_bench(fn)
    return gb / (ms * 1e-3)
 

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Triton kernels")
    parser.add_argument("--all", action = "store_true", help = "Run all benchmarks")
    parser.add_argument("--exp", action = "store_true", help = "Run exponentiation benchmarks")
    parser.add_argument("--log", action = "store_true", help = "Run natural logarithm benchmarks")
    parser.add_argument("--relu", action = "store_true", help = "Run ReLU benchmarks")
    parser.add_argument("--add", action = "store_true", help = "Run addition benchmarks")
    parser.add_argument("--sub", action = "store_true", help = "Run subtraction benchmarks")
    parser.add_argument("--mul", action = "store_true", help = "Run multiplication benchmarks")
    parser.add_argument("--div", action = "store_true", help = "Run division benchmarks")
    parser.add_argument("--matmul", action = "store_true", help = "Run matrix multiplication benchmarks")
    parser.add_argument("--sum", action = "store_true", help = "Run summation benchmarks")
    parser.add_argument("--mean", action = "store_true", help = "Run mean benchmarks")
    parser.add_argument("--var", action = "store_true", help = "Run variance benchmarks")
    parser.add_argument("--std", action = "store_true", help = "Run standard deviation benchmarks")
    parser.add_argument("--emb", action= "store_true", help = "Run embedding benchmarks")
    parser.add_argument("--ln", action = "store_true", help = "Run layer normalization module benchmarks")
    parser.add_argument("--flash", action = "store_true", help = "Run flash attention module benchmarks")
    
    args = parser.parse_args()
    
    # if no args are provided, print help
    if not any(vars(args).values()):
        parser.print_help()
        exit(0)
    
    print(f"Attention:\nBENCHMARK tot_elements ARE DESIGNED TO FUNCTION WITHIN A LIMIT OF 1GB OF GPU MEMORY\n"
          f"IF YOU HAVE LESS YOU WILL GET ERRORS.\nTO FIX, EDIT x_vals IN EACH BENCHMARK'S CONFIG.")
    
    # Generate configs based on selected operation
    unary_ops_args = get_unary_ops_args(args)
    if unary_ops_args:
        print("\nRunning unary operation benchmarks...")
        unary_op_configs.extend(generate_unary_op_configs(unary_ops_args))
        benchmark_unary.run(print_data = True, save_path = "./benchmarks/")
    
    binary_ops_args = get_binary_ops_args(args)
    if binary_ops_args:
        print("\nRunning binary operation benchmarks...")
        binary_op_configs.extend(generate_binary_op_configs(binary_ops_args))
        benchmark_binary.run(print_data = True, save_path = "./benchmarks/")
    
    if args.all or args.matmul:
        print("\nRunning matmul benchmarks...")
        benchmark_matmul.run(print_data = True, save_path = "./benchmarks/")
        
    reduction_args = get_reduction_args(args)
    if reduction_args:
        print("\nRunning reduction benchmarks...")
        reduction_configs.extend(generate_reduction_configs(reduction_args))
        benchmark_reduction.run(print_data = True, save_path = "./benchmarks/")        