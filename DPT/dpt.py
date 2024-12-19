import torch.nn as nn

from .blocks import ReassembleBlock, FeatureFusionBlock

class DPT(nn.Module):
    """Dense Prediction Transformer (Headless)"""
    
    def __init__(self,features,ebed_dim,layer_channels=[256,512,1024,1024]):
        super(DPT,self).__init__()
        
        self.reassemble1 = ReassembleBlock(
            features,ebed_dim,layer_channels[0],
            nn.ConvTranspose2d(
                in_channels=layer_channels[0],
                out_channels=layer_channels[0],
                kernel_size=4,
                stride=4,
                padding=0
            )
        )
        self.reassemble2 = ReassembleBlock(
            features,ebed_dim,layer_channels[1],
            nn.ConvTranspose2d(
                in_channels=layer_channels[1],
                out_channels=layer_channels[1],
                kernel_size=2,
                stride=2,
                padding=0
            )
        )
        self.reassemble3 = ReassembleBlock(
            features,ebed_dim,layer_channels[2],
            nn.Identity()
        )
        self.reassemble4 = ReassembleBlock(
            features,ebed_dim,layer_channels[3],
            nn.Conv2d(
                in_channels=layer_channels[3],
                out_channels=layer_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        
        self.fusion4 = FeatureFusionBlock(features)
        self.fusion3 = FeatureFusionBlock(features)
        self.fusion2 = FeatureFusionBlock(features)
        self.fusion1 = FeatureFusionBlock(features)
        
    def forward(self,x,patches_h,patches_w):
        inter1,inter2,inter3,inter4 = x
        
        assembled1 = self.reassemble1(inter1, patches_h, patches_w)
        assembled2 = self.reassemble2(inter2, patches_h, patches_w)
        assembled3 = self.reassemble3(inter3, patches_h, patches_w)
        assembled4 = self.reassemble4(inter4, patches_h, patches_w)
        
        fused4 = self.fusion4(assembled4, out_size=assembled3.shape[2:])
        fused3 = self.fusion4(fused4, assembled3, out_size=assembled2.shape[2:])
        fused2 = self.fusion4(fused3, assembled2, out_size=assembled1.shape[2:])
        fused1 = self.fusion4(fused2, assembled1)
        
        return fused1
