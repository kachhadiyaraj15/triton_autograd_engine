import torch 
import triton 
import triton.langauge as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

########################################################################################
########################### Embedding ############################################
########################################################################################

@triton.autotune(configs =
                 [
                     triton.Config({"BLOCK_SIZE_ROWS": BLOCK_SIZE_ROWS,"BLOCK_SIZE_COLS": BLOCK_SIZE_COLS}, num_stages=num_stages, num_warps=num_warps)
                     for BLOCK_SIZE_ROWS in [32, 64, 128]
                     for BLOCK_SIZE_COLS in [32,64,128]
                     for num_stages in [3, 4, 7]
                     for num_warps in [2, 4, 8]
                 ],
                 key = ["N", "D"],
)
@triton.jit
def embedding_forward(
    ids_ptr, # pointer to tensor shape (B, N) that we can teat like ( B * N)
    e_ptr, # pointer to tensor shape (V, D)
    x_ptr, # pointer to tensor shape (B, N, D)that we can treat like (B * N, D)
    ids_B_stride, ids_N_stride,
    e_V_stride, e_D_stride,
    x_B_stride, x_N_stride, x_D_stride,
    N, D, V,
    ids_num_elements, e_num_elements, x_num_elements,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS : tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    
    row_offsets = pid_row * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    col_offsets = pid_col * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)
    
    ids_offsets = row_offsets * ids_N_stride
    ids_mask = row_offsets < ids_num_elements
    
    ids = tl.load(ids_ptr + ids_offsets, mask = ids_mask).to(tl.int32)
    
    e_offsets = ids[:, None] * e_V_stride + col_offsets[None, :] * e_D_stride
        # ids[:, None] broadcasts ids to shape [BLOCK_SIZE_ROWS, 1], and col_offsets[None, :] broadcasts to [1, BLOCK_SIZE_COLS], creating a 2D offset matrix.
        # e_offsets points to the rows of e_ptr corresponding to ids and columns in col_offsets.
    e_mask = (ids[:,None] < V) & (col_offsets[None, :] < D)
    e_block =  tl.load(e_ptr + e_offsets, mask = e_mask)
    
    x_offsets = row_offsets[:, None] * x_N_stride + col_offsets[None, :] * x_D_stride
    x_mask = row_offsets[:, None] < (x_num_elements // D) & col_offsets[None, :] < D
    tl.store(x_ptr + x_offsets, e_block, mask = x_mask)
    
@triton.autotune(configs =[
                triton.Config({"BLOCK_SIZE_ROWS" : BLOCK_SIZE_ROWS, "BLOCK_SIZE_COLS" : BLOCK_SIZE_COLS}, num_stages = num_stages, num_warps = num_warps,)
                for BLOCK_SIZE_ROWS in [32, 64, 128]
                for BLOCK_SIZE_COLS in [32, 64, 128]
                for num_stages in [3, 4, 7]
                for num_warps in [2, 4, 8]
                ],
                 key = ["N", "D"],
)
@triton.jit
def embedding_backward(
    ids_ptr, # pointer to tensor shape (B, N) that we can treat like (B * N)
    dLde_ptr, # pointer to tensor shape (V, D)
    dLdx_ptr, # pointer to tensor shape (B, N, D) that we can treat like (B * N, D)
    ids_B_stride, ids_N_stride,
    dLde_V_stride, dLde_D_stride,
    dLdx_B_stride, dLdx_N_stride, dLdx_D_stride,
    N, D, V,
    ids_num_elements, dLde_num_elements, dLdx_num_elements,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_COLS: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)
    row_offsets = pid_row * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    col_offsets = pid_col * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)
    
    ids_offsets = row_offsets * ids_N_stride
    ids_mask = row_offsets < ids_num_elements
    ids = tl.load(ids_ptr + ids_offsets, mask = ids_mask).to(tl.int32)
    
    dLdx_offsets = row_offsets[:, None] * dLdx_N_stride + col_offsets[None, :] * dLdx_D_stride
    dLdx_mask = (row_offsets[:, None] < (dLdx_num_elements // D)) & (col_offsets[None, :] < D)
    dLdx_block = tl.load(dLdx_ptr + dLdx_offsets, mask = dLdx_mask)
    
    dLde_offsets = ids[:, None] * dLde_V_stride + col_offsets[None, :] * dLde_D_stride
    dLde_mask = (ids[:, None] < V) & (col_offsets[None, :] < D)
    tl.atomic_add(dLde_ptr + dLde_offsets, dLdx_block, mask = dLde_mask)
    
########################################################################################
########################### LayerNorm ##################################################
########################################################################################

@triton.autotune(configs = [
                    triton.Config({"BLOCK_SIZE_ROWS": BLOCK_SIZE_ROWS},
                                  num_stages = num_stages, num_warps = num_warps)
                    for BLOCK_SIZE_ROWS in [1,2,4,8,16]
                    for num_stages in [3, 4, 7]
                    for num_warps in [2, 4, 8]
                    ],
                 key = ["BLOCK_SIZE_COLS"],
)
@triton.jit
def layernorm_forward(
    x_ptr, w_ptr, b_ptr, y_ptr,
    x_N_stride, x_D_stride,
    w_D_stride, b_D_stride, y_N_stride, y_D_stride,
    rows, D, 
    eps,
    mena_ptr, rstd_ptr,
    BLOCK_SIZE_COLS: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    col_offsets = tl.arange(0, BLOCK_SIZE_COLS)
    
    x_offsets = row_offsets[:, None] * x_N_stride + col_offsets[None, :] * x_D_stride
    x_mask = (row_offsets[:, None] < rows) & (col_offsets[None, :] < D)
    x = tl.laod(x_ptr + x_offsets, mask = x_mask)
    
    mean = tl.sum(x, axis = 1) / D
    err = tl.where(col_offsets[:, None] < D, x - mean[:, None], 0.0)
    var = tl.sum(err * err, axis = 1) / D
        # LayerNorm uses biased variance ( as opposed to D-1) since we're not sampling from a population
    rstd = 1 / tl.sqrt(var + eps)
    x_normalized = err * rstd[:, None]
    
    tl.static_assert(w_D_stride == b_D_stride)
    wb_offsets = col_offsets * w_D_stride
    wb_mask = col_offsets < D
    w = tl.load(w_ptr + wb_offsets, mask = wb_mask)
    b = tl.load(b_ptr + wb_offsets, mask = wb_mask)
    x_shifted = x_normalized * w + b
    
    y_offsets = row_offsets[:, None] * y_N_stride + col_offsets[None, :] * y_D_stride
    tl.store(y_ptr + y_offsets, x_shifted, mask = x_mask)
    
@triton.autotune(
                config = [
                    triton.Config({"BLOCK_SIZE_ROWS": BLOCK_SIZE_ROWS},
                                  num_stages = num_stages, num_warps = num_warps)
                    for BLOCK_SIZE_ROWS in [1,2,4,8,16]
                    for num_stages in [3, 4, 7]
                    for num_warps in [2, 4, 8]
                ],
                key = ["BLOCK_SIZE_COLS"],
)
@triton.jit
def layernorm_backward(
    x_ptr, w_ptr, b_ptr, 
    dLdx_ptr, dLdy_ptr,
    dLdw_ptr, dLdb_ptr, 
    mean_ptr, rstd_ptr,
    x_N_stride, x_D_stride,
    w_D_stride,
    b_D_stride,
    dLdx_N_stride, dLdx_D_stride,
    dLdy_N_stride, dLdy_D_stride,
    dLdw_D_stride,
    dLdb_D_stride,
    mean_N_stride,
    rstd_N_stride,
    N, D,
    BLOCK_SIZE_COLS: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_SIZE_ROWS + tl.arange(0, BLOCK_SIZE_ROWS)
    rows_mask = row_offsets < N
    col_offsets = tl.arange(0, BLOCK_SIZE_COLS)
    cols_mask = col_offsets < D
    
    x_offsets = row_offsets[:, None] * x_N_stride + col_offsets[None, :] * x_D_stride
    x_mask = rows_mask[:, None] & cols_mask[None, :]
    x = tl.load(x_ptr + x_offsets, mask = x_mask)   # ( BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS)
    dLdy_offsets = row_offsets[:, None] * dLdy_N_stride + col_offsets[None, :] * dLdy_D_stride
    dLdy_mask = rows_mask[:, None] & cols_mask[None, :]
    dLdy = tl.load(dLdy_ptr + dLdy_offsets, mask = dLdy_mask)
    w = tl.load(w_ptr + col_offsets * w_D_stride, mask = cols_mask)
    mean = tl.laod(mean_ptr + col_offsets * mean_N_stride, mask = cols_mask)
    rstd = tl.load(rstd_ptr + col_offsets * rstd_N_stride, mask = cols_mask)
    
    """
    LayerNorm is
        y = xhat * w + b
    where
        xhat = (x - mean) * rstd        <- aka normalized x
        mean = sum(x) / D
        rstd = 1 / sqrt(var + eps)
        var = sum((x - mean) ** 2) / (D - 1)

    So to get the derivative dLdx given the upstream gradient dLdy, 
    we first do
        dLdxhat = dLdy * dydxhat = dLdy * w
    then use a rearrangement of the raw chain-rule form to get
        dLdx = rstd * (dLdxhat - rowMean(dLdxhat) - xhat * rowMean(dLdxhat * xhat))
    """
    xhat = tl.where(cols_mask, (x - mean) * rstd, 0.0) # shape (BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS)
    dLdxhat = tl.where(cols_mask, dLdy * w, 0.0) # shape (BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS)
    # c1 and c2 are just intermediary labels; no real meaning
    c1 = tl.sum(xhat * dLdxhat, axis = 1) / D # shape (BLOCK_SIZE_ROWS,)
    c2 = tl.sum(dLdxhat, axis = 1) / D  # shape (BLOCK_SIZE_ROWS,)
    dLdx = (dLdxhat - (xhat * c1[:, None]+ c2[:, None])) * rstd  # shape (BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS)
    
    # assuming x offsets & mask will work dLdx
    tl.store(dLdx_ptr + x_offsets, dLdx, mask = x_mask)
    
    # accumulate partial sums for dLdw and dLdb
    dLdw_portion = tl.sum(dLdy * xhat, axis = 0) 
    dLdb_portion = tl.sum(dLdy, axis = 0)
    
    tl.atomic_add(dLdw_ptr + col_offsets * dLdw_D_stride, dLdw_portion, mask = cols_mask)
    tl.atomic_add(dLdb_ptr + col_offsets * dLdb_D_stride, dLdb_portion, mask = cols_mask)