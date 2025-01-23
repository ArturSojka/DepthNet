import torch
from torch import nn
import torch.nn.functional as F
from dino import DINOv2
from DPT import DPT
from torchvision.transforms import Compose, Resize, Normalize

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
    
    def __init__(self,backbone=None):
        super(DepthNet,self).__init__()
        
        self.intermediate_layers = [2,5,8,11]
        self.backbone: DINOv2 = backbone if backbone is not None else torch.hub.load(r'C:\Users\artur\.cache\torch\hub\facebookresearch_dinov2_main', 'dinov2_vits14', source='local')
        
        self.dpt = DPT(64,self.backbone.embed_dim,layer_channels=[48, 96, 192, 384])
        
        self.head = DepthHead(64,32)
        
    def forward(self,x):
        size = x.shape[-2:]
        patches_h, patches_w = x.shape[-2] // 14, x.shape[-1] // 14
        
        x = self.backbone.get_intermediate_layers(x, self.intermediate_layers, return_class_token=True)
        
        x = self.dpt(x,patches_h,patches_w)
        
        x = self.head(x, size)
        
        return x.squeeze(1)
    
    @torch.no_grad()
    def infer_image(self,image):
        size = image.shape[:2]
        transformed = self.prepare_for_inference(image,490)
        depth = self.forward(transformed)
        depth = F.interpolate(depth[:, None], size, mode="bilinear", align_corners=True)[0, 0]
        return depth.cpu()
        
    
    def prepare_for_inference(self,image, input_size):
        # new_h, new_w = (image.shape[-2] // 14)*14, (image.shape[-1] // 14)*14
        transform = Compose([
            Resize((input_size,input_size)),
            Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
        ])
        image = image.permute(2,0,1).unsqueeze(0).type(torch.float32) / 255.0
        
        image = transform(image).to('cuda' if torch.cuda.is_available() else 'cpu')
        return image
        
        