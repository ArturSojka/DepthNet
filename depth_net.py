import torch
from torch import nn
import torch.nn.functional as F
from dinov2 import DINOv2
from DPT import DPT

class DepthHead(nn.Module):
    """Depth estimation head"""
    
    def __init__(self, dpt_features, features):
        super(DepthHead,self).__init__()
        
        self.conv1 = nn.Conv2d(dpt_features, dpt_features // 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dpt_features // 2, features, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(features, 1, kernel_size=1, stride=1, padding=0)
        
    def forward(self,x, img_size):
        x = self.conv1(x)
        
        x = F.interpolate(x,size=img_size, mode="bilinear", align_corners=True)
        
        x = self.conv2(x)
        x = self.relu(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        
        return x
        

class DepthNet(nn.Module):
    """Dense Prediction Transformer for monocular depth estimation"""
    
    def __init__(self, backbone_model):
        super(DepthNet,self).__init__()
        
        assert backbone_model in ('dinov2_vits14', 'dinov2_vitb14')
        
        self.intermediate_layers = [2,5,8,11]
        self.backbone: DINOv2 = torch.hub.load('facebookresearch/dinov2', backbone_model)
        
        self.dpt = DPT(256,self.backbone.embed_dim,layer_channels=[256,512,1024,1024])
        
        self.head = DepthHead(256,32)
        
    def forward(self,x):
        size = x.shape[-2:]
        patches_h, patches_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        x = self.backbone.get_intermediate_layers(x, self.intermediate_layers, return_class_token=True)
        
        x = self.dpt(x,patches_h,patches_w)
        
        x = self.head(x, size)
        
        return x