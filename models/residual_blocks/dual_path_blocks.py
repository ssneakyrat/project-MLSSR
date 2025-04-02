import torch
import torch.nn as nn
import torch.nn.functional as F

class DualPathBlock(nn.Module):
    """
    Dual-path block that processes background and foreground features separately.
    The background path uses more aggressive smoothing and global context modeling.
    """
    def __init__(self, in_channels, out_channels):
        super(DualPathBlock, self).__init__()
        
        # Background path (focus on smoothness and homogeneity)
        self.bg_path = nn.Sequential(
            # Larger kernel for capturing smoother patterns
            nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Point-wise conv to mix channel information
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Foreground path (focus on detail)
        self.fg_path = nn.Sequential(
            # Smaller kernel for capturing details
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Attention mechanism to combine paths
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels*2, out_channels*2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels*2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Projection for residual connection if needed
        self.project = None
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x if self.project is None else self.project(x)
        
        # Process through both paths
        bg_features = self.bg_path(x)
        fg_features = self.fg_path(x)
        
        # Concatenate features for attention
        concat_features = torch.cat([bg_features, fg_features], dim=1)
        
        # Apply attention to weight the contribution of each path
        attention_weights = self.attention(concat_features)
        bg_weight = attention_weights[:, 0:1, :, :]
        fg_weight = attention_weights[:, 1:2, :, :]
        
        # Weighted sum of background and foreground features
        out = bg_weight * bg_features + fg_weight * fg_features
        
        # Apply residual connection
        return F.relu(out + residual)

class EncoderBlockDualPath(nn.Module):
    """
    Encoder block for U-Net architecture with dual-path processing.
    
    Each block consists of:
    - Dual-path processing for background and foreground features
    - Residual connection 
    - MaxPool for downsampling
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlockDualPath, self).__init__()
        
        # Dual-path processing
        self.dual_path = DualPathBlock(in_channels, out_channels)
        
        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Process through dual path block
        out = self.dual_path(x)
        
        # Store pre-pooled output for encoder-decoder skip connection
        skip = out
        
        # Apply pooling
        pooled = self.pool(out)
        
        return pooled, skip

class DecoderBlockDualPath(nn.Module):
    """
    Decoder block for U-Net architecture with dual-path processing.
    
    Each block consists of:
    - TransposedConv2D for upsampling
    - Concatenation with skip connection from encoder
    - Dual-path processing for background and foreground features
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlockDualPath, self).__init__()
        
        # Upsampling layer
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        # After concatenation, we have in_channels * 2 channels
        self.conv_reduce = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        
        # Dual-path processing
        self.dual_path = DualPathBlock(in_channels, out_channels)
        
    def forward(self, x, skip):
        # Upsampling
        x = self.up(x)
        
        # Handle potential size mismatches
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)
        
        # Concatenate with skip connection from encoder
        x = torch.cat([x, skip], dim=1)
        
        # Reduce channels
        x = self.conv_reduce(x)
        
        # Process through dual path block
        x = self.dual_path(x)
        
        return x