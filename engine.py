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
