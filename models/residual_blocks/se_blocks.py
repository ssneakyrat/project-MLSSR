import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-Excitation block for adaptive feature recalibration.
    
    Squeezes spatial information into channel-wise statistics and learns
    to selectively emphasize informative features while suppressing less useful ones.
    """
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EncoderBlockResidualSE(nn.Module):
    """
    Enhanced Encoder block with residual connections and Squeeze-Excitation.
    
    Each block consists of:
    - Two Conv2D layers with BatchNorm and ReLU
    - Residual connection from input to after the second conv
    - Squeeze-Excitation block for adaptive feature recalibration
    - MaxPool for downsampling
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super(EncoderBlockResidualSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Squeeze-Excitation block
        self.se = SqueezeExcitationBlock(out_channels, reduction)
        
        # Skip connection with projection if needed
        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        # First conv block
        conv1 = F.relu(self.bn1(self.conv1(x)))
        
        # Second conv block
        conv2 = self.bn2(self.conv2(conv1))
        
        # Apply residual connection
        if self.skip_proj is not None:
            residual = self.skip_proj(x)
        else:
            residual = x
            
        out = F.relu(conv2 + residual)  # Residual connection
        
        # Apply Squeeze-Excitation
        out = self.se(out)
        
        # Store pre-pooled output for encoder-decoder skip connection
        skip = out  
        
        # Apply pooling
        pooled = self.pool(out)
        
        return pooled, skip


class DecoderBlockResidualSE(nn.Module):
    """
    Enhanced Decoder block with residual connections and Squeeze-Excitation.
    
    Each block consists of:
    - TransposedConv2D for upsampling
    - Concatenation with skip connection from encoder
    - First Conv2D to reduce channels
    - Second Conv2D with residual connection
    - Squeeze-Excitation block for adaptive feature recalibration
    """
    def __init__(self, in_channels, out_channels, reduction=16):
        super(DecoderBlockResidualSE, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        # After concatenation, we have in_channels * 2 channels
        self.conv1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Second convolution with residual connection
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Squeeze-Excitation block
        self.se = SqueezeExcitationBlock(out_channels, reduction)
        
        # Skip projection for residual connection
        self.skip_proj = None
        if in_channels != out_channels:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x, skip):
        # Upsampling
        x = self.up(x)
        
        # Handle potential size mismatches
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)
        
        # Concatenate with skip connection from encoder
        x = torch.cat([x, skip], dim=1)
        
        # First convolution to reduce channels
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Store result for residual connection
        residual = x
        
        # Second convolution
        x = self.bn2(self.conv2(x))
        
        # Apply residual connection with projection if needed
        if self.skip_proj is not None:
            residual = self.skip_proj(residual)
            
        x = F.relu(x + residual)
        
        # Apply Squeeze-Excitation
        x = self.se(x)
        
        return x