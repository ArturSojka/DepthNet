import torch
from torch import nn
import torch.nn.functional as F

def invert_depth(true_depth: torch.Tensor):
    return 1.0 / true_depth.clamp(min=1e-6)

def normalize_depth(depth):
    t_d = torch.median(depth)
    s_d = torch.mean(torch.abs(depth - t_d))
    return (depth - t_d) / (s_d + 1e-6)

def ssi_loss(pred_depth, true_depth): # normalized
    return torch.mean(torch.abs(pred_depth - true_depth))