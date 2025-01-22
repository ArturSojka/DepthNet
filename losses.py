import torch
from torch import nn
import torch.nn.functional as F

class GMLoss(nn.Module):
    """
    Gradient Matching Loss as described in https://arxiv.org/pdf/1907.01341v3
    Implementation adapted from https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0
    """
    
    def __init__(self, n_scales):
        super(GMLoss, self).__init__()
        self.n_scales = n_scales
    
    def forward(self, prediction, target):
        total = self.gradient_loss(prediction,target)
        
        for i in range(1,self.n_scales):
            step = pow(2, i)

            total += self.gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step])
            
        return total
    
    def gradient_loss(self, prediction, target):
        diff = prediction - target

        grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
        grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])

        return torch.sum(torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))) / torch.numel(target)

class SSIMSELoss(nn.Module):
    """
    Scale and Shift Invariant Loss as described in https://arxiv.org/pdf/1907.01341v3
    Implementation adapted from https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0
    """
    
    def __init__(self, alpha=0.5):
        super(SSIMSELoss, self).__init__()
        self._alpha = alpha
        self._gm_loss = GMLoss(4)
    
    def forward(self, prediction, target):
        pred_depth = 1/torch.clamp(prediction,min=1e-6)
        true_depth = 1/target
        scale, shift = self.scale_and_shift(pred_depth, true_depth)
        prediction_ssi = scale.view(-1, 1, 1) * pred_depth + shift.view(-1, 1, 1)
        
        res = prediction_ssi - true_depth
        loss = torch.mean(torch.mean(res * res, (1, 2)) / 2)
        
        if self._alpha > 0.0:
            loss += self._alpha * self._gm_loss(1/torch.clamp(prediction_ssi,min=1e-6),target)
            
        return loss
    
    def scale_and_shift(self, prediction, target):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(prediction * prediction, (1, 2))
        a_01 = torch.sum(prediction, (1, 2))
        a_11 = torch.sum(torch.ones_like(target), (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(prediction * target, (1, 2))
        b_1 = torch.sum(target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        valid = det.nonzero()

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1


class MAGELoss(nn.Module):
    """
    Mean Absolute Gradient Error Loss as described in https://arxiv.org/pdf/2410.02073
    """
    def __init__(self, n_scales, device='cpu'):
        super(MAGELoss, self).__init__()
        self.n_scales = n_scales
        self._scharr_x = torch.tensor([[ 3,  0,  -3],
                                       [10,  0, -10],
                                       [ 3,  0,  -3]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        self._scharr_y = torch.tensor([[ 3,  10,   3],
                                       [ 0,   0,   0],
                                       [-3, -10,  -3]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        gaussian_kernel = torch.tensor([
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ], dtype=torch.float32) / 256.0
        
        self._gaussian = gaussian_kernel.unsqueeze(0).unsqueeze(0).to(device)
    
    def forward(self, prediction, target):
        current_pred = prediction.unsqueeze(1)
        current_target = target.unsqueeze(1)
        
        total = self.gradient_loss(current_pred, current_target).squeeze(1)
        
        for _ in range(1, self.n_scales):
            # Apply Gaussian blur before downsampling
            current_pred = F.conv2d(current_pred, self._gaussian, padding=2)
            current_target = F.conv2d(current_target, self._gaussian, padding=2)
            
            # Downsample blurred images
            current_pred = F.interpolate(current_pred, scale_factor=0.5, mode='bilinear', align_corners=False)
            current_target = F.interpolate(current_target, scale_factor=0.5, mode='bilinear', align_corners=False)
            
            total += self.gradient_loss(current_pred, current_target).squeeze(1)
        
        return torch.mean(total / self.n_scales)
    
    def gradient_loss(self, prediction, target):
        grad_prediction_x = F.conv2d(prediction, self._scharr_x, padding=0)
        grad_prediction_y = F.conv2d(prediction, self._scharr_y, padding=0)
        grad_target_x = F.conv2d(target, self._scharr_x, padding=0)
        grad_target_y = F.conv2d(target, self._scharr_y, padding=0)
        
        mag_prediction = torch.sqrt(grad_prediction_x ** 2 + grad_prediction_y ** 2)
        mag_target = torch.sqrt(grad_target_x ** 2 + grad_target_y ** 2)
        
        return torch.mean(torch.abs(mag_prediction - mag_target), (2,3))

class SSIMAELoss(nn.Module):
    """
    Robust Scale and Shift Invariant Loss as described in https://arxiv.org/pdf/1907.01341v3
    Simmilar to the loss used in https://arxiv.org/pdf/2410.02073
    """
    
    def __init__(self,alpha=0.5,device='cpu'):
        super(SSIMAELoss, self).__init__()
        self._alpha = alpha
        self._gm_loss = MAGELoss(4,device)
    
    def forward(self, prediction, target):
        scale_p, shift_p, scale_t, shift_t = self.scale_and_shift(prediction, target)
        prediction_ssi = (prediction - shift_p.view(-1, 1, 1)) / scale_p.view(-1, 1, 1)
        target_ssi = (target - shift_t.view(-1, 1, 1)) / scale_t.view(-1, 1, 1)
        
        res = torch.abs(prediction_ssi - target_ssi)
        loss = torch.mean(torch.mean(res, (1, 2)) / 2)
        
        if self._alpha > 0.0:
            loss += self._alpha * self._gm_loss(prediction_ssi,target_ssi)
            
        return loss
    
    def scale_and_shift(self, prediction, target):
        shift_p = torch.median(prediction.view(prediction.size(0),-1),dim=1).values
        shift_t = torch.median(target.view(target.size(0),-1),dim=1).values
        
        scale_p = torch.mean(torch.abs(prediction - shift_p.view(-1,1,1)), (1,2))
        scale_t = torch.mean(torch.abs(target - shift_t.view(-1,1,1)), (1,2))
        return (scale_p, shift_p, scale_t, shift_t)
    