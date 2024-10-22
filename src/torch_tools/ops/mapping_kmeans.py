import torch
from torch import Tensor, Size
from typing import Union
from .. import _C


def mapping_dist_324(src: Tensor, index: Tensor, dist_shape: Union[list[int], Size]) -> torch.Tensor:
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

    if index.dtype != torch.int32:
        index = index.type(torch.int32)

    shape_src = torch.tensor(src.shape, dtype=torch.int32).to(device=src.device)
    shape_index = torch.tensor(index.shape, dtype=torch.int32).to(device=index.device)
    shape_dist = torch.tensor(dist_shape, dtype=torch.int32).to(device=src.device)

    dist = _C.mapping_dist_324(src, index, shape_src, shape_index, shape_dist)

    del shape_src, shape_index, shape_dist

    return dist.type(dist_type)

def mapping_src_223(src: Tensor, index: Tensor, dist_shape: Union[list[int], Size]) -> torch.Tensor:
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

    if index.dtype != torch.int32:
        index = index.type(torch.int32)

    shape_src = torch.tensor(src.shape, dtype=torch.int32).to(device=src.device)
    shape_index = torch.tensor(index.shape, dtype=torch.int32).to(device=index.device)
    shape_dist = torch.tensor(dist_shape, dtype=torch.int32).to(device=src.device)

    dist = _C.mapping_src_223(src, index, shape_src, shape_index, shape_dist)

    del shape_src, shape_index, shape_dist

    return dist.type(dist_type)

def mapping(src: Tensor, index: Tensor, dist_shape: Union[list[int], Size]) -> torch.Tensor:
    """
    Arguments:
        src (`torch.Tensor`): Source tensor to map

        index (`torch.Tensor`): Index tensor which has same shape as *src*'s high dims, \
            and their elements maps sencond dim from *src* to *dist*

        dist_shape (`torch.Tensor`): Shapes of *dist* 
    """
    if len(src.shape) == 3 and len(index.shape) == 2 and len(dist_shape) == 4:
        return mapping_dist_324(src, index, dist_shape)
    
    if len(src.shape) == 2 and len(index.shape) == 2 and len(dist_shape) == 3:
        return mapping_src_223(src, index, dist_shape)
    
    raise NotImplementedError("unsupported mapping dim-len group: {}-{}-{}".format(len(src.shape), len(index.shape), len(dist_shape)))
