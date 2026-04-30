import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(pred, target, alpha=2, beta=4):
    """
    Modified focal loss for heatmap prediction.
    pred: [B, C, H, W] - probabilities (after sigmoid)
    target: [B, C, H, W] - ground truth heatmap [0, 1]
    """
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()

    neg_weights = torch.pow(1 - target, beta)
    
    # Clip predictions to prevent log(0)
    pred = torch.clamp(pred, 1e-4, 1 - 1e-4)
    
    loss = 0
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class JointLoss(nn.Module):
    """
    Multi-task loss for ST-TAF Net processing both Classification and Detection tasks.
    $L = \lambda_1 L_{cls} + \lambda_2 L_{det}$
    $L_{det} = L_{heatmap} + L_{offset} + L_{size}$
    """
    def __init__(self, lambda_cls=1.0, lambda_det=1.0, lambda_hm=1.0, lambda_off=1.0, lambda_size=0.1):
        super(JointLoss, self).__init__()
        self.lambda_cls = lambda_cls
        self.lambda_det = lambda_det
        
        self.lambda_hm = lambda_hm
        self.lambda_off = lambda_off
        self.lambda_size = lambda_size
        
        self.cls_criterion = nn.CrossEntropyLoss()
        
    def forward(self, preds_cls, heatmaps, offsets, sizes, target_cls, target_heatmaps, target_offsets, target_sizes, mask):
        """
        preds_cls: [B, num_event_classes]
        heatmaps: [B, num_det_classes, H, W]
        offsets: [B, 2, H, W]
        sizes: [B, 2, H, W]
        
        target_cls: [B] integers
        target_heatmaps: [B, num_det_classes, H, W]
        target_offsets: [B, 2, H, W]
        target_sizes: [B, 2, H, W]
        mask: [B, H, W] binary mask indicating where object centers are located
        """
        # 1. Classification Loss (Cross Entropy)
        l_cls = self.cls_criterion(preds_cls, target_cls)
        
        # 2. Detection Loss
        # Heatmap focal loss
        l_hm = focal_loss(heatmaps, target_heatmaps)
        
        # Valid mask expansion for L1 losses (only compute offset/size loss at object centers)
        # mask is [B, H, W], expand to [B, 2, H, W]
        mask_expanded = mask.unsqueeze(1).expand_as(offsets)
        num_objects = mask.sum().clamp(min=1.0) # avoid division by zero
        
        # Offset L1 loss
        l_off = F.l1_loss(offsets * mask_expanded, target_offsets * mask_expanded, reduction='sum') / num_objects
        
        # Size L1 Loss
        l_size = F.l1_loss(sizes * mask_expanded, target_sizes * mask_expanded, reduction='sum') / num_objects
        
        l_det = (self.lambda_hm * l_hm) + (self.lambda_off * l_off) + (self.lambda_size * l_size)
        
        # Combine Loss
        total_loss = (self.lambda_cls * l_cls) + (self.lambda_det * l_det)
        
        return total_loss, l_cls, l_det, l_hm, l_off, l_size
