import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

"""
this implementation of flash-attention only supports a causal mask, no other masks or lack of a mask
the forward pass is based primarily on the pseudocode from the two original papers
https://arxiv.org/abs/2205.14135
https://arxiv.org/abs/2307.08691
and the backward passs is based primarily on the triton documentation implementation since it's 
significantly faster than the pseudocode from the original papers
https://triton-lang.org/main/getting-started/tutorials/06-fused-attentionhtml#sphx-glr-getting-started-tutorials-06-fused-attention-py
"""

@triton.jit
def _attn_fwd_inner(
    Q, O, L, M,
    K_ptr, V_ptr, 
    K_T_offsets, V_offsets,
    block_index_QO,
    softmax_scale,
    stride_K_N, stride_V_N,
    BLOCK_SIZE_QO : tl.constexpr, BLOCK_SIZE_KV : tl.constexpr,
    DIAGONAL: tl.constexpr, # whether this is a diagonal block or not
    offsets_QO_N : tl.constexpr, offsets_KV_N : tl.constexpr,
    N : tl.constexpr, Dh : tl.constexpr,
):
    # example to understand the below code :
    """
    Suppose N=256 N = 256 N=256, BLOCK_SIZE_QO = 64, BLOCK_SIZE_KV = 64, and block_index_QO = 2.
    - The query block covers indices i=[128,129,…,191] i = [128, 129, \ldots, 191] i=[128,129,…,191].
    - **First call (DIAGONAL=False)**:
        - Key range: lo=0, hi=128, covering key blocks 0 and 1 (indices j=[0,1,…,127] j = [0, 1, \ldots, 127] j=[0,1,…,127]).
        - These are below-diagonal blocks, so no masking is applied (all j≤i j \leq i j≤i).
    - **Second call (DIAGONAL=True)**:
        - Key range: lo=128, hi=192, covering key block 2 (indices j=[128,129,…,191] j = [128, 129, \ldots, 191] j=[128,129,…,191]).
        - This is the on-diagonal block, so the causal mask is applied to ensure only j≤i j \leq i j≤i elements are unmasked.
    """
    if DIAGONAL:
        # used only for blocks along the diagonal in which there is transition between non-masked and masked keys
        lo = block_index_QO * BLOCK_SIZE_QO
        hi = (block_index_QO + 1) * BLOCK_SIZE_QO
        # let the compiler know lo is  amultiple of BLOCK_SIZE_QO to speed things up
        lo = tl.multiple(lo, BLOCK_SIZE_QO) # TODO not sure why this doen't also help with hi
    else:
        # this part is for any blocks in the causal mask below the diagonal
        lo, hi = 0, block_index_QO * BLOCK_SIZE_QO
        
    K_T_offsets += lo * stride_K_N
    V_offsets += lo * stride_V_N
    offsets_KV_N += lo
    
    # loop over blocks along the N dimensions of K and V update the O accumulator whie=le doing so 
    for start_KV in range(lo, hi, BLOCK_SIZE_KV):
        strat_KV = tl.multiple_of(start_KV, BLOCK_SIZE_KV)
        # when in doubbt, use tl.multiple_of() for any dynamic variables(as opposed to static variables)
        
        # compute ( Q @ K^T) / sqrt(Dh)
        N_mask_KV = offsets_KV_N < N
        K_T = tl.load(K_ptr + K_T_offsets, mask = N_mask_KV[None, :], other = 0.) # shape (Dh, BLOCK_SIZE_KV)
            # sequence mask sets non-existent tokens in the block past N to zero vector
        S = tl.dot(Q, K_T) * softmax_scale # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
            # the masked tokens create columns & rows of zeros hugging the bottom and right edges of S
        
        if DIAGONAL:# if we're currently on a block containing the diagonal
            # the causal mask is True on the lower-traingle including the diagonal
            causal_mask = offsets_QO_N[:, None] >= (offsets_KV_N[None, :])
            # causal mask addition sets upper-traingle values (excluding diagonal) to -inf
            S + tl.where(causal_mask, 0, -1.0e6) # shape (BLOCK_SIZE_QO, BLOCK_SIZE_KV)
        # notice that the masked out tokens previously hugging the right edeg of S have all been replaced with -inf and the masked out tokens hugging the bottom edge are still mostly 0's but with some -infs towsrds the right edge of each of them, except for the last one which is only 0's
        
        # find the max values of the new block and compare them to those of all previous blocks to get an update 
        M_new = tl.maximum(M, tl.amx(S, axis = 1)) # shape (BLOCK_SIZE_QO)
            # masked token rows ar the bottom will return a maximum value of 0 since their only values are 0 and -inf 
        # adjust S block for safe softmax computation
        S -= M_new[:, None]
        # in the case of masked non-existent tokens that means substracting by 0 so on difference
        
        # Compute the exponentials of each safe dot product, which will be the numerator of our softmax
        P = tl.exp2(S)
            # we're using base 2 instead of base e bcz it's faster and softamx is invariant to the change, however it dows make the derivative in the backward pas a bit more complicates.
            # for the masked non-existent tokens as the bottom that will be 2^0 = 1 for all those entries
        
        # compute the sum by rows of the attention scores
        L_new = tl.sum(P, axis = 1) # shape (BLOCK_SIZE_QO)
            # for tha masked non-existent tokens we're summing a bunch of 1's with some -infs, except for the very bottom one which is just 1's and therefor its sum is the largest being equal to BLOCK_SIZE_QO 
        # this alpha is the correction factor we'll use on the previous L
        alpha = tl.exp2( M - M_new) # shape(BLOCK_SIZE_QO)
            # for the masked non-existent tokens that's just 2^(1-1) = 1 = alpha_i so no correction 
        # apply the correction factor to the previous L and add new L
        L = L * alpha + L_new # shape (BLOCK_SIZE_QO)
            # for each of the masked non-existent tokens they approach N for their enry L_i
        
        # This computes O = P @ V + O * alpha
        V = tl.load(V_ptr + V_offsets, mask = N_mask_KV[:, None], other = 0.) # shape (BLOCK_SIZE_KV, Dh)
        O = O * alpha[:, None] # adjusts previous values based on potential new max
        # accumulated P and V block dot product into O
        O = tl.dot(P, V, acc = O) # shape (BLOCK_SIZE_QO, Dh)
            # notice we're ding this V projection before we've actually divided by out softmax denominator l_i which is possible because in this context the two operations are associative
            # acc tells triton to accumulate the vaues into O_block
            # the masked non-existent tokens are a bunch of 1's in the bottom rows of P and 0's in the bottom rows of V. This matmul leaves O with a bunch of incorrect values in its bottom rows, but they will get ignored later when we store O with a proper mask
        
        M = M_new # update the running maximum for the next block
        # iterate pointers 
        K_T_offsets += BLOCK_SIZE_KV * stride_K_N
        V_offsets += BLOCK_SIZE_KV * stride_V_N
        offsets_KV_N += BLOCK_SIZE_KV
    return O, L, M # we save these three specifically for use later in the backward pass


@triton.autotune( # decorator figures out what meta-parameters will be most efficient
                 [
                     triton.Config(
                         {"BLOCK_SIZE_QO" : BLOCK_SIZE_QO, "BLOCK_SIZE_KV" : BLOCK_SIZE_KV},
                         num_stages = num_stages, num_warps = num_warps,
                     )
                        for BLOCK_SIZE_QO in [32, 64, 128]
                        for BLOCK_SIZE_KV in [32, 64, 128]
                        for num_stages in [3, 4, 7]
                        for num_warps in [2, 4, 8]
                        if BLOCK_SIZE_QO  == BLOCK_SIZE_KV # They should only be one hyperparameter then
                 ],
                    key = ["N", "D"],  # auto-tune will re-run every time either of these values changes in a new input
)
@triton.jit
def attn_fwd(
    Q_ptr, K_ptr, V_ptr,  # each shape (B,H,N,Dh)
    O_ptr,  # shape (B,H,N,Dh), store final output
    LSE_ptr,  # shape (B,H,N), first store the max values of each row & later the logsumexp trick
    sofmax_scale,
    stride_Q_B, stride_Q_H, stride_Q_N, stride_Q_Dh,
    stride_K_B, stride_K_H, stride_K_N, stride_K_Dh,
    stride_V_B, stride_V_H, stride_V_N, stride_V_Dh,
    stride_O_B, stride_O_H, stride_O_N, stride_O_Dh,
    stride_LSE_B, stride_LSE_H, stride_LSE_N,
    B, # unlike other tensor dimensions, batch size needs to be flexible for runtime differneces
    H:tl.constexpr, # number of heads, fixed at compile time
    N:tl.constexpr, # sequence length, fixed at compile time
    Dh:tl.constexpr, # hidden size, fixed at compile time, should always be a power of 2, and really 128 and 256 are the only reasonble options
    BLOCK_SIZE_QO: tl.constexpr, # size of the blocks for Q and O, fixed at compile time
    BLOCK_SIZE_KV: tl.constexpr, # size of the blocks for K and V, fixed at compile time
):
    # in order to use tl.exp2 later instead of tl.exp (the former is faster) we need to scale our softmax scale by ln2
    rln2: tl.constexpr = 1.4426950408889634
    softmax_scale *= rln2
    
    # as opposed to regular assert, static_assert occurs at compile-time
    tl.static_assert(BLOCK_SIZE_KV <= Dh)
    
    # This indicates which block in the sequence length to process
    block_index_QO = tl.program_id(0)
    # This indicates which head and batch to process. Each program is associated with a single head of a single batch
    index_BH = tl.program_id(1)
    # This indicates which batch this program is associated with (each batch has H heads)
    index_B = index_BH // H
    # This indicates the position of the head in the batch
    index_H = index_BH % H
    
    # This allows to get shape (N, Dh) block in the  Q, K, V, and o by indexing it by batch and head
    Q_ptr += index_B * stride_Q_B + index_H * stride_Q_H
    K_ptr += index_B * stride_K_B + index_H * stride_K_H
    V_ptr += index_B * stride_V_B + index_H * stride_V_H
    O_ptr += index_B * stride_O_B + index_H * stride_O_H
    
    # offsets for N are split by pids but for Dh we keep the whole thing in SRAM
    offsets_QO_N = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO)
    offsets_KV_N = tl.arange(0, BLOCK_SIZE_KV)
    offsets_Dh = tl.arange(0, Dh)
    
    # Create offsets specific to each tensor
    Q_offsets = (offsets_QO_N[:, None] * stride_Q_N + offsets_Dh[None, :] * stride_Q_Dh)
        # shape : (BLOCK_SIZE_QO, Dh)
    # Transpose K while loading it (as oppeses to writing a whole separate kernel for transpose)
    K_T_offsets = (offsets_Dh[:, None] * stride_V_Dh + offsets_KV_N[None, :] * stride_V_N)
        # shape : (Dh, BLOCK_SIZE_KV)
    V_offsets = (offsets_KV_N[:, None] * stride_V_N + offsets_Dh[None, :] * stride_V_Dh)
        # shape: (BLOCK_SIZE_KV, Dh)
    
    # load the block of Q that this PID will use: it will stay in SRAM throughout the inner loop
    N_mask_QO = offsets_QO_N < N
    Q = tl.load(Q_ptr + Q_offsets, mask = N_mask_QO[:, None],other = 0.) # shape (BLOCK_SIZE_QO, Dh)
        # Sequence mask sets non-existent tokens in the block past N to zero vector
    
    ## Pre-allocate tensors for storing intermediate & output values
    # the running maximum. we have one entry for each query in the block we're currently working on 
    M = tl.full(shape = [BLOCK_SIZE_QO], value = -1e6, dtype = tl.float32) # large negative number will get ignore by tl.max()
    # the running sum. We hae one entry for each query ( since we sum the attention scores by rows)
    L = tl.full(shape= [BLOCK_SIZE_QO], value = 1.0, dtype = tl.float32) # this is the sum of the exponentials of the attention scores,, 1 is because we'll using exponentials and e^0 = 1
    # the accumulator for the output, hich is a group of rows of the matrix 
    O = tl.zeros([BLOCK_SIZE_QO, Dh], dtype = tl.float32) 
    
    # calculate attention for dense blocks (Those where the mask if full of 1's)
    # This step runs for the blocks below the diagonal in causal attention
    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        softmax_scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        False, # blocks on the DIAGONAL get special treatement if this is set to true; we use it below
        offsets_QO_N, offsets_KV_N,
        N, Dh,
    )    
    
    # This map runs for the blocks on the diagonal in the causal attention mask 
    O, L, M = _attn_fwd_inner(
        Q, O, L, M,
        K_ptr, V_ptr,
        K_T_offsets, V_offsets,
        block_index_QO,
        softmax_scale,
        stride_K_N, stride_V_N,
        BLOCK_SIZE_QO, BLOCK_SIZE_KV,
        True, # blocks on the diagonal get special masking treatment
        offsets_QO_N, offsets_KV_N,
        N, Dh,
    )
    
    # finally dividing by the denominator of i=our softmax
    # notice we;ve already multiplied by V to get O, so this was done out-of-order from naive softmax implementations
    O = O / L[:, None] # shape (BLOCK_SIZE_QO, Dh) / (BLOCK_SIZE_QO, 1) = (BLOCK_SIZE_QO, Dh) 
        # we can do this out-of-order since the matmul (tl.dot in _attn_fwd_inner) and this eentry-wise division are associative. matmul and entry-wise-ops are not normally, but at this level of granularity it's no longer actually a matmul but instead individiaul dot-products
        # the maskded non-existent tokens are a bunch of meanigless values in the bottom rows of O and generally roughly equal to N in the bottom entries of L. Dividing the former by the latter isn't going to breaa=k anything and we'll mask them out later when storing 
    
    # this is needed to compute the logsumep (LSE) for the backwards pass. basically instead of saving the maxes and the sums separately, we save them together which still works thanks to exponential aithmetic 
    LSW = M + tl.math.log2(L) # L was composed using the sum & exp operations in _attn_fwd_inner()
        # this will work because softmax(x_i) = exp(x_i - m_i) / l_i
        #                                     = exp(x_i - m_i) / exp(log(l_i))
        #                                     = exp(x_i - m_i - log(l_i))
        # the masked non-existent tokens are a bunch of 0's in the bottom entries of M and a bunch of values roughly equal to N in the bottom entris of L. So in LSE they'll be a bunch of llod_2(N) entries at the bottom that we of course don't plan to use
    
    # storing it all back to DRAM
    LSE_offsets = index_BH * stride_LSE_H + offsets_QO_N
    LSE_mask = block_index_QO * BLOCK_SIZE_QO + tl.arange(0, BLOCK_SIZE_QO) < N
    tl.store(LSE_ptr + LSE_offsets, LSW, mask = LSE_mask)
        # the mask prevents us from saving the useless log_2(N) values at the bottom of LSE
    O_offsets = (offsets_QO_N[:, None] * stride_O_N + offsets_Dh[None, :] * stride_O_Dh)
        # shape : (BLOCK_SIZE_QO, Dh)
    tl.store(O_ptr + O_offsets, O, mask = N_mask_QO[:, None])
        # the mask prevents us from saving the useless values at the bottom of O corresponding to non-existent tokens
    
    
    