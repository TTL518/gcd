import torch
import torch.nn as nn
from torch.nn import functional as F


class GlobalDescriptor(nn.Module):
    '''
    Generalized pooling operator as defined in the original paper.

    Args:
        p (int/float): Parameter that determines the pooling operator.
    Returns:
        The pooled tensor.
    '''
    def __init__(self,
                 p=1):
        super(GlobalDescriptor, self).__init__()
        self.p = p

    def forward(self, x):
        assert x.dim() == 4, 'The input tensor of GlobalDescriptor must be of shape [B, C, H, W]'
        if self.p == 1:
            return x.mean(dim=[-1, -2])
        elif self.p == float('inf'):
            return torch.flatten(F.adaptive_max_pool2d(x, output_size=(1, 1)), start_dim=1)
        else:
            # Equation
            sum_value = x.pow(self.p).mean(dim=[-1, -2])
            return torch.sign(sum_value) * (torch.abs(sum_value).pow(1.0 / self.p))


class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()

    def forward(self, x):
        assert x.dim() == 2, 'The input tensor of L2Norm must be of shape [B, C]'
        return F.normalize(x, p=2, dim=-1)
