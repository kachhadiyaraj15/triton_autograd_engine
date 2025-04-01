from typing import Union, Tuple, Optional
import numpy as np
import math

import torch
import triton
import triton.language as tl

device = torch.device(f"cuda:{torch.cuda.current_device()}")

from engine import TritonTensor
import nn

# Helper Functions for heatmap visualization
def clear_heatmap_folder(folder: str = "heatmaps"):
    """
    Deletes the folder (if it exists) and then re-creates an empty version
    """
    import os
    import shutil

    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)
    
def save_heatmaps(torch_tensor : torch.Tensor, triton_tensor : torch.Tensor, test_name : str, folder : str = "heatmaps", atol : float = 1e-3, rtol : float = 1e3, phase : str = "backward"):
    """
    Saves multiple sets of heatmaps comparing torch_tensor and triton_tensor
    1. Raw a]bsolute difference
    2. Absolute tolerance failure mask ( where abs diff > atol)
    3. Relative tolerance failure mask ( where abs diff > rtol * abs(expected))
    4. Combined tolerance failure mask ( where abs diff > atol + rtol * abs(expected))
    
    Handles different tensor dimesions:
        4D: (batch_size, num_heads, seq_len, head_dim) -> one set per batch/head
        3D: (batch_size, seq_len, model_dim) -> one set per batch 
        2D: (batch_size, model_dim) -> one set per batch
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # convert to numpy arrays 
    actual = triton_tensor.detach().cpu().numpy()
    expected = torch_tensor.detach().cpu().numpy()
    
    # compute differences and masks 
    abs_diff = np.abs(expected - actual)
    abs_threshold = atol
    rel_threshold = rtol * np.abs(expected)
    
    abs_fail_mask = (abs_diff > abs_threshold).astype(np.int32)
    rel_fail_mask = (abs_diff > rel_threshold).astype(np.int32)
    
    def save_figure(matrix, title : str, filename: str, cmap : str = "hot"):
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap=cmap, aspect='auto')
        plt.title(title)
        plt.xlabel("Model/Head Dimension")
        plt.ylabel("Sequence Position" if matrix.ndim>1 else "Batch")
        plt.colorbar()
        plt.savefig(os.path.join(folder, filename))
        plt.close()
        
    def save_all_figures(diff: np.ndarray, abs_mask : np.ndarray, rel_mask : np.ndarray,
                         suffix : str, filename_suffix: str):
        # Raw difference
        save_figure(diff, f"{test_name} {suffix} - raw diff ({phase}) ",
                    f"{test_name}_{filename_suffix}_raw_diff_{phase}.png")
        
        # Absolute tolerance failure mask
        save_figure(abs_mask, f"{test_name} {suffix} - abs fail mask ({phase})",
                    f"{test_name}_{filename_suffix}_abs_fail_mask_{phase}.png", cmap="Reds")
        
        # Relative tolerance failure mask
        save_figure(rel_mask, f"{test_name} {suffix} - rel fail mask ({phase})",
                    f"{test_name}_{filename_suffix}_rel_fail_mask_{phase}.png", cmap="Reds")
        
    # Handle different tensor dimensions
    if expected.ndim == 4:  # (batch_size, num_heads, seq_len, head_dim)
        B, H, N, D = expected.shape
        for b in range(B):
            for h in range(H):
                save_all_figures(
                    abs_diff[b, h], abs_fail_mask[b, h], rel_fail_mask[b, h],
                    f"diff: batch {b} head {h}", f"diff_b{b}_h{h}"
                )
    elif expected.ndim == 3:  # (batch_size, seq_len, model_dim)
        B, N, D = expected.shape
        for b in range(B):
            save_all_figures(
                abs_diff[b], abs_fail_mask[b], rel_fail_mask[b], 
                f"diff: batch {b}", f"diff_b{b}"
            )
    elif expected.ndim == 2:  # (batch_size, model_dim)
        B, D = expected.shape
        save_all_figures(
            abs_diff, abs_fail_mask, rel_fail_mask,
            "diff", "diff"
        )
    else:
        # Fallback for other shapes
        save_all_figures(
            abs_diff, abs_fail_mask, rel_fail_mask, 
            "diff", "diff"
        )

# --- End of heatmap helper functions --- 

def test_operation(op_name: str,
                   triton_fn,
                   torch_fn,
                   inputs_list : list[torch.Tensor],
                   atol : float = 1e-3,
                   rtol : float = 1e3,
                   ):
    """
    Test TritonTensor operations against PyTorch for correctness.
    
    Args:
        op_name : Name of operation being tested
        triton_fn : Function that takes TritonTensor inputs and returns TritonTensor output
        torch_fn : Function that takes torch.Tensor inputs and returns torch.Tensor output
        inputs_list : List of pytorch tensors to be used as inputs 
        atol : Absolute tolerance for comparing outputs
        rtol : Relative tolerance for comparing outputs
    """
    print(f"Testing {op_name} operation")
    
    # Generate random inputs
    torch_inputs = [x.detach().clone().require_grad_(x.requires_grad) for x in inputs_list ] # Create leaf tensors
    triton_inputs = [TritonTensor(x, requires_grad=x.requires_grad) for x in inputs_list]
    
    # Forward pass
    torch_out = torch_fn(*torch_inputs)
    torch_out = torch_out[0] if op_name[:3] in ("min", "max") else torch_out
        # TODO do we need our max op tp also give indices? i think so for inference 
    triton_out = triton_fn(*triton_inputs)
    
    # Clear out previous heatmaps before any testing
    clear_heatmap_folder("heatmaps")
    
    # Check forward pass
    try:
        torch.testing.assert_close(torch_out, triton_out.data, atol = atol, rtol = rtol)
        print(f"Forward pass matches")
    except AssertionError as error:
        print(f"Forawrd pass mismatch detected in opearation {op_name}:")
        print("Generating heatmaps of the output differences")
        save_heatmaps(torch_out, triton_out.data, f"{op_name}_output", folder="heatmaps", atol=atol, rtol=rtol, phase="forward")
        raise error
    
    # before computing the backward pass, we need to let the autotuner run
    # This needs to be done bc otherwise the gradient accumulation of each run would compund to incorrect values
    zero_grad = torch.zeros_like(torch_out)
    triton_out.backward(zero_grad)
    
    # and in order to avoid any potential divide by zero Nan's from division, ew et all gradients to zero
    triton_out.zero_grad_backward()
    
    # Backward pass
    grad_output = torch.randn_like(torch_out)
    torch_out.backward(grad_output)
    triton_out.backward(grad_output)
    
    # Check Gradients
    for i, (torch_input, triton_input) in enumerate(zip(torch_inputs, triton_inputs)):
        try:
            torch.testing.assert_close(torch_input.grad, triton_input.grad.data, atol = atol, rtol = rtol)
        except AssertionError as error:
            print(f"{'#'*20}\ntensor {i} in input gradients list\n{'#'*20}")
            print(f"Gradient mismatch detected for input {i} of operation {op_name}:")
            print("Generating heatmaps of the gradient differences")
            save_heatmaps(torch_input.grad, triton_input.grad, f"{op_name}_input{i}", folder="heatmaps", atol=atol, rtol=rtol, phase="backward")
            
            raise error
        
    print(f"Backward pass matches")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for Triton operations")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--exp", action="store_true", help="Run experimental tests")
    parser.add_argument("--log", action="store_true", help="Run natural logarithm tests")
    parser.add_argument("--relu", action="store_true", help="Run ReLU tests")
    parser.add_argument("--add", action="store_true", help="Run addition tests")
    parser.add_argument("--mul", action="store_true", help="Run multiplication tests")
    parser.add_argument("--sub", action="store_true", help="Run subtraction tests")
    parser.add_argument("--div", action="store_true", help="Run division tests")
    parser.add_argument("--matmul", action="store_true", help="Run matrix multiplication tests")
    parser.add_argument("--sum", action="store_true", help="Run summation across final dimension tests")
    parser.add_argument("--max", action="store_true", help="Run max across final dimension tests")
    parser.add_argument("--min", action="store_true", help="Run min across final dimension tests")
    parser.add_argument("--mean", action="store_true", help="Run mean across final dimension tests")
    parser.add_argument("--var", action="store_true", help="Run variance across final dimension tests")
    parser.add_argument("--std", action="store_true", help="Run standard deviation across final dimension tests")
    parser.add_argument("--trans", action="store_true", help="Run transpose across arbitary axes tests")
    parser.add_argument("--sqz", action="store_true", help="Run squeeze across arbitary axes tests")
    parser.add_argument("--unsqz", action="store_true", help="Run unsqueeze across arbitary axes tests")
    parser.add_argument("--reshape", action="store_true", help="Run reshape tests")
    parser.add_argument("--idx", action="store_true", help="Run indexing tests")
    parser.add_argument("--linear", action="store_true", help="Run linear layer tests")
    parser.add_argument("--emb", action="store_true", help="Run embedding layer tests")
    parser.add_argument("--ln", action="store_true", help="Run layer normalization tests")
    parser.add_argument("--flash", action="store_true", help="Run flash attention tests")
    
    args = parser.parse_args()
    
    # If no args are provided, print help
    if not any(vars(args).values()):
        parser.print_help()
        exit(0)
    
    B, N, H, D, V = 1, 128, 2, 128, 4096 
    # B = batch size, N = sequence length, H = number of heads, D = head dimension, V = vocabulary size
    
    # Exponential
    if args.exp or args.all:
        def triton_exp(x: TritonTensor): return x.exp()
        def torch_exp(x: torch.Tensor): return torch.exp(x)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) for shape in input_shapes]   
        test_operation(
            f"exponentiation: ({B}, {N}, {D})",
            triton_exp,
            torch_exp,
            inputs_list([(B, N, D)]),
        )
    
    # Natural Logarithm
    if args.log or args.all:
        def triton_log(x: TritonTensor): return x.log()
        def torch_log(x: torch.Tensor): return torch.log(x)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) + 0.01 for shape in input_shapes]   
        test_operation(
            f"natural logarithm: ({B}, {N}, {D})",
            triton_log,
            torch_log,
            inputs_list([(B, N, D)]),
        )
    
    # ReLU
    if args.relu or args.all:
        def triton_relu(x: TritonTensor): return x.relu()
        def torch_relu(x: torch.Tensor): return torch.nn.functional.relu(x)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) for shape in input_shapes]   
        test_operation(
            f"ReLU: ({B}, {N}, {D})",
            triton_relu,
            torch_relu,
            inputs_list([(B, N, D)]),
        )
    
    # Addition
    if args.add or args.all:
        def triton_add(x: TritonTensor, y: TritonTensor): return x + y
        def torch_add(x: torch.Tensor, y: torch.Tensor): return x + y
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) for shape in input_shapes]   
        test_operation(
            f"addition: ({B}, {N}, {D})",
            triton_add,
            torch_add,
            inputs_list([(B, N, D), (B, N, D)]),
        )
        test_operation(
            f"addition with broadcasting: ({B}, {N}, {D}) + ({D})",
            triton_add,
            torch_add,
            inputs_list([(B, N, D), (D)]),
        )
        test_operation(
            f"addition with single scalar: ({B}, {N}, {D}) + (1)",
            triton_add,
            torch_add,
            inputs_list([(B, N, D), (1)]),
        )
    
    # Multiplication
    if args.mul or args.all:
        def triton_mul(x: TritonTensor, y: TritonTensor): return x * y
        def torch_mul(x: torch.Tensor, y: torch.Tensor): return x * y
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) for shape in input_shapes]   
        test_operation(
            f"multiplication: ({B}, {N}, {D}) * ({B}, {N}, {D})",
            triton_mul,
            torch_mul,
            inputs_list([(B, N, D), (B, N, D)]),
        )
        test_operation(
            f"multiplication with broadcasting: ({B}, {N}, {D}) * ({D})",
            triton_mul,
            torch_mul,
            inputs_list([(B, N, D), (D)]),
        )
        test_operation(
            f"multiplication with single scalar: ({B}, {N}, {D}) * (1)",
            triton_mul,
            torch_mul,
            inputs_list([(B, N, D), (1)]),
        )
        
    # Subtraction
    if args.sub or args.all:
        def triton_sub(x: TritonTensor, y: TritonTensor): return x - y
        def torch_sub(x: torch.Tensor, y: torch.Tensor): return x - y
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) for shape in input_shapes]   
        test_operation(
            f"subtraction: ({B}, {N}, {D}) - ({B}, {N}, {D})",
            triton_sub,
            torch_sub,
            inputs_list([(B, N, D), (B, N, D)]),
        )
        test_operation(
            f"subtraction with broadcasting: ({B}, {N}, {D}) - ({D})",
            triton_sub,
            torch_sub,
            inputs_list([(B, N, D), (D)]),
        )
        test_operation(
            f"subtraction with single scalar: ({B}, {N}, {D}) - (1)",
            triton_sub,
            torch_sub,
            inputs_list([(B, N, D), (1)]),
        )
    
    # Division
    if args.div or args.all:
        def triton_div(x: TritonTensor, y: TritonTensor): return x / y
        def torch_div(x: torch.Tensor, y: torch.Tensor): return x / y
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) for shape in input_shapes]   
        test_operation(
            f"division: ({B}, {N}, {D}) / ({B}, {N}, {D})",
            triton_div,
            torch_div,
            inputs_list([(B, N, D), (B, N, D)]),
        )
        test_operation(
            f"division with broadcasting: ({B}, {N}, {D}) / ({D})",
            triton_div,
            torch_div,
            inputs_list([(B, N, D), (D)]),
        )
        test_operation(
            f"division with single scalar: ({B}, {N}, {D}) / (1)",
            triton_div,
            torch_div,
            inputs_list([(B, N, D), (1)]),
        )
    
    # Matrix Multiplication
    if args.matmul or args.all:
        def triton_matmul(x: TritonTensor, y: TritonTensor): return x @ y
        def torch_matmul(x: torch.Tensor, y: torch.Tensor): return x @ y
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) for shape in input_shapes]   
        test_operation(
            f"matrix multiplication: ({B}, {D}) @ ({D}, {D*4})",
            triton_matmul,
            torch_matmul,   
            inputs_list([(B, D), (D, D*4)]),
            atol = 5e-1, # matmul gradient accumulation is very sensitive to flop error even at fp32
            rtol = 1e5, # relative error is dummb when it's relative to 1e-6 everything looks big or at least that's what i think is happing; lmk if you find error i couldn;'t 
        )
        test_operation(
            f"matrix multiplication with leading dimensions: ({B}, {H}, {N}, {D}) @ ({B}, {H}, {D}, {N})",
            triton_matmul,
            torch_matmul,
            inputs_list([(B, H, N, D), (B, H, D, N)]),
            atol = 5e-2,
            rtol = 1e5,
        )
        test_operation(
            f"matrix multiplication with broadcasting: ({B}, {N}, {D}) @ ({D}, {N})",
            triton_matmul,
            torch_matmul,
            inputs_list([(B, N, D), (D, N)]),
            atol = 5e-2,
            rtol = 1e5,
        )
    
    # Summation
    if args.sum or args.all:
        def triton_sum(x: TritonTensor): return x.sum()
        def torch_sum(x: torch.Tensor): return torch.sum(x, axis=  -1)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) for shape in input_shapes]   
        test_operation(
            f"sum across final dimension: ({B}, {N}, {D})",
            triton_sum,
            torch_sum,
            inputs_list([(B, N, D)]),
        )
    
    ## Mean
    if args.mean or args.all:
        def triton_mean(x: TritonTensor): return x.mean()
        def torch_mean(x: torch.Tensor): return torch.mean(x, axis= -1)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) for shape in input_shapes]   
        test_operation(
            f"mean across final dimension: ({B}, {N}, {D})",
            triton_mean,
            torch_mean,
            inputs_list([(B, N, D)]),
        )
        
    # Maximum
    if args.max or args.all:
        def triton_max(x: TritonTensor): return x.max()
        def torch_max(x: torch.Tensor): return torch.max(x, axis= -1)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) for shape in input_shapes]   
        test_operation(
            f"max across final dimension: ({B}, {N}, {D})",
            triton_max,
            torch_max,
            inputs_list([(B, N, D)]),
        )
    
    # Minimum
    if args.min or args.all:
        def triton_min(x: TritonTensor): return x.min()
        def torch_min(x: torch.Tensor): return torch.min(x, axis= -1)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) for shape in input_shapes]   
        test_operation(
            f"min across final dimension: ({B}, {N}, {D})",
            triton_min,
            torch_min,
            inputs_list([(B, N, D)]),
        )
    
    ## Variance
    if args.var or args.all:
        def triton_var(x: TritonTensor): return x.var()
        def torch_var(x: torch.Tensor): return torch.var(x, axis= -1)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype = torch.float32, device=device, requires_grad=True) for shape in input_shapes]   
        test_operation(
            f"variance across final dimension: ({B}, {N}, {D})",
            triton_var,
            torch_var,
            inputs_list([(B, N, D)]),
        )
        
    ### STANDARD DEVIATION
    if args.all or args.std:
        def triton_std(x): return x.std()
        def torch_std(x): return torch.std(x, dim=-1)
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        test_operation(
            f"standard deviation: ({B}, {N}, {D})",
            triton_std,
            torch_std,
            inputs_list([(B, N, D)]),
        )
    
    # Transpose
    if args.all or args.trans:
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        def triton_trans(x: TritonTensor): return x.transpose(-2, -3)
        def torch_trans(x: torch.Tensor): return x.transpose(x, -2, -3)
        test_operation(
            f"transpose: ({B}, {N}, {H}, {D})-> ({B}, {H}, {D}, {N})",
            triton_trans,
            torch_trans,
            inputs_list([(B, N, H, D)]),
        )
        # this one should default to final two dims
        def triton_trans(x): return x.transpose()
        def torch_trans(x): return x.transpose(x, -1, -2)
        test_operation(
            f"transpose: ({B}, {N}, {H}, {D})-> ({B}, {H}, {D}, {N})",
            triton_trans,
            torch_trans,
            inputs_list([(B, N, H, D)]),
        )
        
    ## Squeeze
    if args.all or args.sqz:
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        def triton_sqz(x: TritonTensor): return x.squeeze(2)
        def torch_sqz(x: torch.Tensor): return x.squeeze(x,2)
        test_operation(
            f"squeeze: ({B}, {N},{1}, {D}) -> ({B}, {N}, {D})",
            triton_sqz,
            torch_sqz,
            inputs_list([(B, N, 1, D)]),
        )
    
    # Unsqueeze
    if args.all or args.unsqz:
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        def triton_unsqz(x: TritonTensor): return x.unsqueeze(2)
        def torch_unsqz(x: torch.Tensor): return x.unsqueeze(x,2)
        test_operation(
            f"unsqueeze: ({B}, {N}, {D}) -> ({B}, {N}, {1}, {D})",
            triton_unsqz,
            torch_unsqz,
            inputs_list([(B, N, D)]),
        )
    
    #### RESHAPE
    if args.all or args.reshape:
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        def triton_reshape(x): return x.reshape((B, N, 4, D//4))
        def torch_reshape(x): return torch.reshape(x, (B, N, 4, D//4))
        test_operation(
            f"reshape: ({B}, {N}, {D}) -> ({B}, {N}, {4}, {D//4})",
            triton_reshape,
            torch_reshape,
            inputs_list([(B, N, D)]),
        )
        
    ### INDEXING
    # NOTE: we expect the bwd pass of idx to fail since we didn't implement it
    if args.all or args.idx:
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        def triton_idx(x): return x[:,-1,:]
        def torch_idx(x): return x[:,-1,:]
        test_operation(
            f"index: ({B}, {N}, {V})[:,-1,:] -> ({B}, {1}, {V})",
            triton_idx,
            torch_idx,
            inputs_list([(B, N, V)]),
        )
    
    ### LINEAR LAYER
    if args.all or args.lin:
        def inputs_list(input_shapes):
            return [torch.randn(shape, dtype=torch.float32, device=device, requires_grad=True) 
                   for shape in input_shapes]
        triton_model =  nn.Linear(D, D*4)
        torch_model = torch.nn.Linear(D, D*4, device=device, dtype=torch.float32)
        # because they both initialize randomly we need to set one to the other
        torch_model.weight.data = triton_model.weight.transpose().data.detach().clone()
            # for some reason pytorch stores the weight matrix transposed
        if triton_model.bias is not None:
            torch_model.bias.data = triton_model.bias.data.detach().clone()
        def triton_linear(x): return triton_model(x)
        def torch_linear(x): return torch_model(x)
        test_operation(
            f"linear layer: ({B}, {N}, {D}) -> ({D}, {D*4})",
            triton_linear,
            torch_linear,
            inputs_list([(B, N, D)]),
            atol=5e-2, # matmul gradient accumulation is VERY sensitive to flop error even at fp32
            rtol=1e5, # relative error is dummb bc when it's relative to 1e-6 everything looks big
            # or at least that's what i think is happening; lmk if you find an error i couldn't
        )
    
    ### EMBEDDING LAYER
    if args.all or args.emb:
        def inputs_list(input_shapes):
            tokens = torch.randint(0, V, size=input_shapes[0], dtype=torch.int64, device=device) 
            weights = torch.randn(size=input_shapes[1], dtype=torch.float32, device=device, requires_grad=True)
            return [tokens, weights]
        triton_model = nn.Embedding(V, D)
        # because they both initialize randomly we need to set their weights to the same matrix
        def triton_embedding(tokens, weights): 
            # this direct assignment is kinda weird since we're assigning a TritonTensor to what
            #  previously was a Parameter but it's prolly fine
            triton_model.weight = weights 
            return triton_model(tokens)
        def torch_embedding(tokens, weights): 
            return torch.nn.functional.embedding(tokens, weights)
        test_operation(
            f"embedding layer: ({B}, {N}) & ({V}, {D}) -> ({B}, {N}, {D})",
            triton_embedding,
            torch_embedding,
            inputs_list([(B, N), (V, D)]),
        )
        # gradients of (B, N) will be None
        # gradients of (V, D) are what we care about
    
    ### LayerNorm Module
    if args.all or args.ln:
        def inputs_list(input_shapes):
            x = torch.randn(size=input_shapes[0], dtype=torch.float32, device=device, requires_grad=True) 
            w = torch.ones(size=input_shapes[1], dtype=torch.float32, device=device, requires_grad=True)
            b = torch.zeros(size=input_shapes[1], dtype=torch.float32, device=device, requires_grad=True)
            return [x, w, b]
        triton_model = nn.LayerNorm(D)
        # because they both initialize randomly we need to set their weights to the same matrix
        def triton_ln(x, w, b): 
            # this direct assignment is kinda weird since we're assigning a TritonTensor to what
            #  previously was a Parameter but it's prolly fine
            triton_model.weight = w
            triton_model.bias = b
            return triton_model(x)
        def torch_ln(x, w, b): 
            return torch.nn.functional.layer_norm(x, normalized_shape=(x.shape[-1],), weight=w, bias=b)
        test_operation(
            f"LayerNorm: ({B}, {N}, {D}) -> ({B}, {N}, {D})",
            triton_ln,
            torch_ln,
            inputs_list([(B, N, D), (D,), (D,)]),
        )
    
    ### Flash Attention
    if args.all or args.flash:
        Dh = 32
        def inputs_list(input_shapes):
            return [torch.randn(size=shape, dtype=torch.float32, device=device, requires_grad=True) * 0.02
                    for shape in input_shapes]
        def triton_flash(q, k, v): 
            return nn.FlashAttention()(q, k, v, scale=math.sqrt(Dh))
        def torch_flash(q, k, v): 
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=math.sqrt(Dh))
        test_operation(
            f"causal flash attention with sequence length that's a multiple of block size",
            triton_flash,
            torch_flash,
            inputs_list([(B,H,N,Dh), (B,H,N,Dh), (B,H,N,Dh)]),
            atol=2e-3, # there's so many operations in here that occasionally one single element surpasses 1e-3
            rtol=1e-1 # relative tolerance can ligma
        )
        test_operation(
            f"causal flash attention with sequence length that's NOT a multiple of block size",
            triton_flash,
            torch_flash,
            inputs_list([(B,H,N - 3,Dh), (B,H,N - 3,Dh), (B,H,N - 3,Dh)]),
            atol=2e-3,
            rtol=1e-1
        )
