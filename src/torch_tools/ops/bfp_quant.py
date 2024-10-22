import torch
from torch import Tensor
from typing import Optional
from .. import _C

def BFPquant(
        src: Tensor,
        exp: Tensor,
        emin: float,
        emax: float,
        mbit: float,
        stochastic: bool,
        max_number:Optional[float] = None,
        ) -> torch.Tensor:
    """
    Arguments:
        src (`torch.Tensor`): Source tensor to map

        index (`torch.Tensor`): Index tensor which has same shape as *src*'s high dims, \
            and their elements maps sencond dim from *src* to *dist*

        dist_shape (`torch.Tensor`): Shapes of *dist* 
    """
    dist_type = src.dtype

    if src.dtype != torch.float32:
        src = src.type(torch.float32)

    if exp.dtype != torch.float32:
        exp = exp.type(torch.float32)

    if max_number is None:
        max_number: float = 2**(emax)*(2-2**(-mbit))

    res = _C.bfp_quant(src, exp, emin, emax, max_number, mbit, stochastic)
    return res.to(dist_type)