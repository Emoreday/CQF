from torch import Tensor

def mapping_dist_324(src: Tensor, index: Tensor, shape_src: Tensor, shape_index: Tensor, shape_dist: Tensor) -> Tensor:
    """
    Arguments:
        src (`torch.Tensor`): Source tensor to map

        index (`torch.Tensor`): Index tensor which has same shape as *src*'s high dims, \
            and their elements maps sencond dim from *src* to *dist*

        dist (`torch.Tensor`): Dist tensor to be mapped

        shape_* (`torch.Tensor`): Shapes of *src*, *index* and *dist* 
    """

def mapping_src_223(src: Tensor, index: Tensor, shape_src: Tensor, shape_index: Tensor, shape_dist: Tensor) -> Tensor: ...

def bfp_quant(src: Tensor, exp: Tensor, emin: float, emax: float, max_number: float, mbit: float, stochastic: bool) -> Tensor: ...