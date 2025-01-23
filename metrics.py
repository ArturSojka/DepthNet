import torch
import os
import cv2
from losses import SSIMSELoss

def compute_abs_rel(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)
        
    abs_diff = torch.abs(pred[mask] - target[mask])
    
    rel_abs_diff = abs_diff / target[mask]
    
    return torch.mean(rel_abs_diff)

def compute_delta1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, threshold: float = 1.25) -> torch.Tensor:
    if mask is None:
        mask = torch.ones_like(pred, dtype=torch.bool)
    
    ratio1 = pred[mask] / target[mask]
    ratio2 = target[mask] / pred[mask]
    
    max_ratio = torch.max(ratio1, ratio2)
    
    correct_pixels = torch.sum(max_ratio < threshold).float()
    total_pixels = torch.sum(mask).float()
    
    return correct_pixels / total_pixels

def compute_metrics(model):
    ibms = r"C:\Users\artur\Downloads\ibims1_core_raw"
    images = os.listdir(os.path.join(ibms,"rgb"))
    loss = SSIMSELoss(0.0)
    abs_rel = 0.0
    delta1 = 0.0
    for i, image_path in enumerate(images):
        if not image_path.endswith("png") or image_path == "livingroom_14.png" or image_path == "office_04.png" or image_path == "storageroom_08.png":
            continue
        im = cv2.imread(os.path.join(ibms,"rgb",image_path))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_tensor = torch.tensor(im)
        
        invalid = cv2.imread(os.path.join(ibms,"mask_invalid",image_path),cv2.IMREAD_GRAYSCALE)
        transparent = cv2.imread(os.path.join(ibms,"mask_transp",image_path),cv2.IMREAD_GRAYSCALE)
        
        depth = cv2.imread(os.path.join(ibms,"depth",image_path),cv2.IMREAD_ANYDEPTH)
        depth_tensor = torch.tensor(depth)*65535.0/50.0
        depth_tensor = (depth_tensor - depth_tensor.min()) / (depth_tensor.max() - depth_tensor.min())
        mask = torch.tensor(invalid*transparent,dtype=torch.bool)*(depth_tensor>=1e-5)
        
        output = model.infer_image(im_tensor)
        # output = (output - output.min()) / (output.max() - output.min())
        
        scale, shift = loss.scale_and_shift(output.unsqueeze(0), depth_tensor.unsqueeze(0))
        output_ssi = (scale.view(-1, 1, 1) * output.unsqueeze(0) + shift.view(-1, 1, 1)).squeeze(0)
        
        abs_rel += compute_abs_rel(output_ssi,depth_tensor,mask)
        delta1 += compute_delta1(output_ssi,depth_tensor,mask)
        if torch.isnan(abs_rel).item() or torch.isinf(abs_rel).item():
            print(i, image_path, abs_rel)
            print(depth_tensor)
            print(output)
            return
    
    return abs_rel / len(images), delta1 / len(images)
        