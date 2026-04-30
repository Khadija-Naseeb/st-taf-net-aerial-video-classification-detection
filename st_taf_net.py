import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSEModule(nn.Module):
    """
    Temporal Squeeze-and-Excitation Module.
    Re-calibrates channel features throughout the video clip.
    """
    def __init__(self, channels, reduction=16):
        super(TemporalSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class WSE_AVT_Block(nn.Module):
    """
    Window-based Squeeze-and-Excitation Aerial Video Transformer Block.
    A simplified version utilizing 3D convolution as a proxy for Window-based Attention
    coupled with Temporal SE as described in the ST-TAF Net context to model 5D tensors.
    """
    def __init__(self, in_channels, out_channels):
        super(WSE_AVT_Block, self).__init__()
        # Proxy for Window-based Attention using 3D spatial-temporal convolution
        self.attn_proxy = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm = nn.LayerNorm(out_channels)
        self.se = TemporalSEModule(out_channels)

    def forward(self, x):
        res = x
        x = self.attn_proxy(x)
        # Apply layer norm over channels: [B, C, T, H, W] -> transpose for LayerNorm -> transpose back
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.relu(x)
        x = self.se(x)
        if res.size() == x.size():
            x = x + res
        return x

class AnchorFreeHead(nn.Module):
    """
    Anchor-Free Detection Head predicting Heatmap, Offset, and Size.
    """
    def __init__(self, in_channels, num_classes):
        super(AnchorFreeHead, self).__init__()
        # Common feature extraction per frame
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1)
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1)
        )

    def forward(self, x):
        # x is a 2D spatial feature map [B, C, H, W]
        feat = F.relu(self.conv1(x))
        heatmap = torch.sigmoid(self.heatmap_head(feat))
        offset = self.offset_head(feat)
        size = F.relu(self.size_head(feat))
        return heatmap, offset, size

class ST_TAF_Net(nn.Module):
    """
    Spatio-Temporal Transformer-based Anchor-Free Network
    """
    def __init__(self, in_channels=3, event_classes=20, detection_classes=10):
        super(ST_TAF_Net, self).__init__()
        # Initial feature projection
        self.proj = nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        
        # WSE-AVT Backbone Blocks
        self.block1 = WSE_AVT_Block(64, 128)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.block2 = WSE_AVT_Block(128, 256)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Classification branch
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.cls_head = nn.Linear(256, event_classes)

        # Detection branch
        self.det_head = AnchorFreeHead(in_channels=256, num_classes=detection_classes)

    def forward(self, x):
        """
        x: [B, C, T, H, W]
        """
        x = F.relu(self.proj(x))
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        encoded_features = self.pool2(x) # Shape: [B, 256, T', H', W']
        
        # 1. Global Event Classification
        # GAP over spatial and temporal dimensions
        global_feat = self.global_pool(encoded_features).view(encoded_features.size(0), -1)
        event_cls = self.cls_head(global_feat)

        # 2. Anchor-Free Detection
        # To perform object detection, we usually aggregate over the temporal dimension
        # or evaluate frame-by-frame. The paper suggests the detection uses the same map.
        # We'll average the temporal dimension to provide an aggregated spatial map
        # Alternatively, we could process a specific key frame.
        spatial_features = torch.mean(encoded_features, dim=2) # Shape: [B, 256, H', W']
        heatmap, offset, size = self.det_head(spatial_features)

        return event_cls, heatmap, offset, size

# Example usage/test
if __name__ == "__main__":
    b, c, t, h, w = 2, 3, 8, 256, 256
    model = ST_TAF_Net()
    x = torch.randn(b, c, t, h, w)
    event_cls, heatmap, offset, size = model(x)
    print("Event Classification shape:", event_cls.shape)
    print("Heatmap shape:", heatmap.shape)
    print("Offset shape:", offset.shape)
    print("Size shape:", size.shape)
