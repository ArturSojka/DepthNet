# code adapted from https://github.com/isl-org/DPT/blob/main/dpt/blocks.py

import torch
import torch.nn as nn


class ResidualConvUnit(nn.Module):
    """Residual convolution unit."""

    def __init__(self, features):
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion module."""

    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()

        self.resConvUnit1 = ResidualConvUnit(features)
        self.resConvUnit2 = ResidualConvUnit(features)
        
        self.out_conv = nn.Conv2d(
            features, features, kernel_size=1, stride=1, padding=0, bias=True,
        )

    def forward(self, *xs, out_size=None):
        output = xs[0]

        if len(xs) == 2:
            output += self.resConvUnit1(xs[1])

        output = self.resConvUnit2(output)

        resize = {"size": out_size} if out_size is not None else {"scale_factor": 2}

        output = nn.functional.interpolate(
            output, **resize, mode="bilinear", align_corners=True
        )
        
        output = self.out_conv(output)

        return output

class Mlp(nn.Module):
    
    def __init__(self,features):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(features * 2, features)
        self.ac1 = nn.GELU()
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.ac1(x)
        
        return x

class ReassembleBlock(nn.Module):
    """Reassemble module"""
    
    def __init__(self,features,in_channels,out_channels,resize_layer):
        super(ReassembleBlock, self).__init__()
        
        self.mlp = Mlp(in_channels)
        self.project = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.resize = resize_layer
        self.resample = nn.Conv2d(out_channels,features, kernel_size=3, stride=1, padding=1, bias=False)
        
        
    def forward(self, x, h_patches, w_patches):
        
        x, readout_token = x[0], x[1]
        
        # Read
        readout_token = readout_token.unsqueeze(1).expand_as(x)
        x = self.mlp(torch.cat((x, readout_token), -1))
        
        # Concatenate
        x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], h_patches, w_patches))
        
        # Project
        x = self.project(x)

        # Resample
        x = self.resize(x)
        x = self.resample(x)
        
        return x