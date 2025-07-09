import torch
import torch.nn as nn
from ._gridencoder import (
    grid_encode_forward,
    grid_encode_backward,
    grad_total_variation,
)

class GridEncoder(torch.nn.Module):
    def __init__(self,
                 input_dim,
                 num_levels,
                 level_dim,
                 base_resolution,
                 log2_hashmap_size,
                 desired_resolution,
                 gridtype="hash",
                 align_corners=False):
        super().__init__()

        self.input_dim = input_dim
        self.num_levels = num_levels
        self.level_dim = level_dim
        self.base_resolution = base_resolution
        self.log2_hashmap_size = log2_hashmap_size
        self.desired_resolution = desired_resolution
        self.gridtype = gridtype
        self.align_corners = align_corners

        # This is the required attribute
        self.output_dim = num_levels * level_dim


    def forward(self, *args, **kwargs):
        return grid_encode_forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        return grid_encode_backward(*args, **kwargs)

    def total_variation(self, *args, **kwargs):
        return grad_total_variation(*args, **kwargs)
