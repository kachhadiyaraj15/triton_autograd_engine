import torch
import triton
import triton.language as tl

DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")

@triton.autotune( # decorator figures out what meta-parameters will be most efficient
                 [
                     triton.Config({"BLOCk_SIZE": BLOCK_SIZE}, num_stages=num_stages, num_warps=num_warps,)
                     for BLOCK_SIZE in [32, 64, 128, 256, 512, 1024]
                        for num_stages in [1, 2, 3, 4]
                        for num_warps in [1, 2, 4, 8]
                 ],
                 key = ["n_elements"], # auto-tune will re-run every time this value is different in a new input 
)
@triton.jit
def unary_op_forward(
    x_ptr,     # pointer to input tensor
    y_ptr,    # pointer to output tensor
    n_elements, # number of elements in the tensor
    op : tl.constexpr, 
    BLOCK_SIZE : tl.constexpr, # number of threads per block
):
    pid = tl.program_id(0) 
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    if op == "exp":
        tl.store(y_ptr + offsets, tl.exp(x), mask=mask)
    if op == "log":
        tl.store(y_ptr + offsets, tl.log(x), mask=mask)
    if op == "relu":
        tl.store(y_ptr + offsets, tl.clamp(x, 0.0, 1e6), mask=mask)   # tl.clamp(x, min_val, max_val) = min(max(x, min_val), max_val)
        
@triton.autotune( # decorator figures out what meta-parameters will be most efficient
                 [
                     triton.Config({"BLOCk_SIZE": BLOCK_SIZE}, num_stages=num_stages, num_warps=num_warps,)
                     for BLOCK_SIZE in [32, 64, 128, 256, 512, 1024]
                        for num_stages in [1, 2, 3, 4]
                        for num_warps in [1, 2, 4, 8]
                 ],
                 key = ["n_elements"], # auto-tune will re-run every time this value is different in a new input 
)
@triton.jit
def unary_op_backward(
    x_ptr,     # pointer to input tensor
    dx_ptr,    # pointer to gradient of input
    z_ptr,     # pointer to the forward pass' output tensor
    dz_ptr,    # pointer to the incoming gradient
    n_elements, # number of elements in the tensor
    op : tl.constexpr,
    BLOCK_SIZE : tl.constexpr, # number of threads per block
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    dz = tl.load(dz_ptr + offsets, mask=mask)
    
    if op == "exp":
        z = tl.load(z_ptr + offsets, mask=mask)
        tl.store(dx_ptr + offsets, dz * z, mask=mask)
    if op == "log":
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(dx_ptr + offsets, dz / x, mask=mask)
    if op == "relu":
        z = tl.load(z_ptr + offsets, mask=mask)
        tl.store(dx_ptr + offsets, dz * (z > 0.0), mask=mask)

@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config({"BLOCK_SIZE": BLOCK_SIZE}, num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE in [32, 64, 128, 256, 512, 1024, 2048, 4096] # values chosen by totally guessing
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["n_elements", "loop_stride"], # auto-tune will re-run every time either of these values are different in a new input
)
@triton.jit # this decorator tells Triton to compile this function into GPU code
def binary_op_forward(
    x_ptr, y_ptr,               # pointers to input vectors (triton converts torch.tensor objects into pointers to their first element)
    output_ptr,                 # ptr to output vector
    n_elements,                 # size of x tensor
    loop_stride,                # size of y tensor
    OP: tl.constexpr,           # known at compile-time so a different kernel gets created for every operation
    BLOCK_SIZE: tl.constexpr,   # number of elements each program should process
):
    # tl.constexpr ia a type that tells the compier that the value must be known at compile-time(not runtime)
    # there are multiple "programs" processing data ( a program is a unique instantiation of this kernel)
    # programs can be defined along multiple dimensions when the inputs have multiple dimesions
    # this op is 1D so axis = 0 is te only option, but bigger operations later may define program_id as a tuple
    # here we identify program we are:
    program_id = tl.program_id(0) 
        # Each program instance gets a unique ID along the specified axis
        # For a vector of length 256 and BLOCK_SIZE=64:
        # program_id=0 processes elements [0:64]
        # program_id=1 processes elements [64:128]
        # program_id=2 processes elements [128:192]
        # program_id=3 processes elements [192:256]
    
    # This program will process inputs that are offset from the initial data(^ described above )
    # note that offsets is a list of pointers a la [0, 1, 2, 3, ..., 62, 63]
    block_start_x = program_id * BLOCK_SIZE
    block_start_y = block_start_x % loop_stride # this is the offset for the second input (y)
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE) # offsets for the first input (x)
    offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE)
    
    # create a mask to guard memeory operations against going out of bounds
    mask_x = offsets_x < n_elements
    mask_y = offsets_y < loop_stride
    
    # load x and y from DRAM ( global GPU memory) into SRAM (on-chip memory)
    # SRAM is much faster but limited in size
    # These masks ensure we don't access memory beyond the tensors' ends
    x = tl.load(x_ptr + offsets_x, mask=mask_x)
    y = tl.load(y_ptr + offsets_y, mask=mask_y)
    
    # perform the operation for this block on SRAM
    # triton has its own internal definitions of all the basic ops that deal with the actual entry-wise details 
    # the conditional here is on a compile-time constant 
    # which Triton can "fold" or "inline" so there's no runtime overhead
    # basically, you'll get a separate compiled kernel per value of OP.
    if OP == "add":
        out = x + y
    elif OP == "sub":
        out = x - y
    elif OP == "div":
        out = x / y
    elif OP == "mul":
        out = x * y
    else:
        raise ValueError(f"input operation must be either 'add', 'sub', 'div', or 'mul' but isntead got {OP}")

    # write back to DRAM, being sure to mask to avoid out-of-bounds accesses
    tl.store(output_ptr + offsets_x, out, mask = mask_x)
    
@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config({"BLOCK_SIZE": BLOCK_SIZE}, num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE in [32, 64, 128, 256, 512, 1024, 2048, 4096] # values chosen by totally guessing
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["n_elements", "loop_stride"], # auto-tune will re-run every time either of these values are different in a new input
)
@triton.jit
def binary_op_backward_dx(
    y_ptr,               
    dx_ptr,             
    do_ptr,                     # pointer to incoming gradient
    n_elements,                 # total number of elements in x and output tensors
    loop_stride,                # total number of elements in y tensor
    OP: tl.constexpr,           # known at compile-time so a different kernel gets created for every operation
    BLOCK_SIZE: tl.constexpr,   # number of elements each program should process
):
    # Get program ID
    pid = tl.program_id(axis=0)

    # calculate starting offset for this program instance
    block_start_x = pid * BLOCK_SIZE
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE) # offsets for the first input (x)
    
    # Create masks to guard memory operations
    mask_x = offsets_x < n_elements
    
    # Load incoming gradient & target gradient
    do = tl.load(do_ptr + offsets_x, mask=mask_x)
    dx = tl.load(dx_ptr + offsets_x, mask=mask_x)
    
    # why are we accumulating dx?
    if OP == "add":
        dx += do
    elif OP == "sub":
        dx += do
    else:
        # prep for mul or div 
        block_start_y = block_start_x % loop_stride # the looping is how we handle broadcasting
        offsets_y = block_start_y + tl.arange(0, BLOCK_SIZE)
        mask_y = offsets_y < loop_stride
        y_val = tl.load(y_ptr + offsets_y, mask=mask_y)
        
        if OP == "mul":
            dx += do * y_val
        
        if OP == "div":
            dx += do / y_val
        
    tl.store(dx_ptr + offsets_x, dx, mask=mask_x) 
    
@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config({"BLOCK_SIZE": BLOCK_SIZE}, num_stages=num_stages, num_warps=num_warps,)
        for BLOCK_SIZE in [32, 64, 128, 256, 512, 1024, 2048, 4096] # values chosen by totally guessing
        for num_stages in ([3, 4, 7])
        for num_warps in [2, 4, 8]
    ],
    key=["n_elements", "loop_stride"], # auto-tune will re-run every time either of these values are different in a new input
)
@triton.jit
def binary_op_backward_dy(
    x_ptr, y_ptr,               # pointers to input vectors
    dy_ptr,             # pointer to each input's gradient, or None if y doesn't require a gradient
    do_ptr,                     # pointer to incoming gradient
    n_elements,                 # total number of elements in x and output tensors
    loop_stride,                # total number of elements in y tensor
    OP: tl.constexpr,           # known at compile-time so a different kernel gets created for every operation
    BLOCK_SIZE: tl.constexpr,   # number of elements each program should process
):
    # Get program ID
    pid = tl.program_id(axis=0)

    # calculate starting offset for this program instance
    block_start_x = pid * BLOCK_SIZE
    block_start_y = block_start_x % loop_stride # the looping is how we handle broadcasting
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE)
    offsets_y = (block_start_y + tl.arange(0, BLOCK_SIZE)) % loop_stride
    
    # Creates masks to guard memory operations
    mask_x = offsets_x < n_elements
    mask_y = offsets_y < loop_stride
    
    # load incoming gradient do
    do = tl.load(do_ptr + offsets_x, mask = mask_x)
    
    # if we were to use same code as the kernel above, then in the case of broadcasting the threads would be overwriting each other. Instead, we use atomic_add which uses a locking mechanism to ensure that only one thread works on a given entry in dy at a time.
    if OP == "add":
        tl.atomic_add(dy_ptr + offsets_y, do, mask = mask_y)
        
    elif OP == "sub":
        tl.atomic_add(dy_ptr + offsets_y, -do, mask = mask_y)
    
    else:
        # prep for mul and div
        x_val = tl.laod(x_ptr + offsets_x, mask = mask_x)
        
        if OP == "mul":
            # dy = do * x
            tl.atomic_add(dy_ptr + offsets_y, x_val * do, mask = mask_y)
        elif OP == "div":
            y_val = tl.load(y_ptr + offsets_y, mask = mask_y)
            # out = x / y => dy = - (x * do) / y^2
            tl.atomic_add(dy_ptr + offsets_y, - (x_val * do) / (y_val * y_val), mask = mask_y)