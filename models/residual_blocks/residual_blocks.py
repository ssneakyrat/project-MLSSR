import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckResidual(nn.Module):
    """
    Bottleneck block between encoder and decoder with residual connection.
    
    The bottleneck transforms:
    in_channels -> out_channels -> in_channels
    
    The residual connection skips from input to output (both with in_channels).
    """
    def __init__(self, in_channels, out_channels):
        super(BottleneckResidual, self).__init__()
        # First conv: in_channels -> out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second conv: out_channels -> in_channels (back to original)
        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
        # No skip projection needed because input and output have same channels
        # We're going from in_channels -> out_channels -> in_channels
        
    def forward(self, x):
        # Store input for residual connection
        residual = x
        
        # First convolution block (down)
        x = self.relu1(self.bn1(self.conv1(x)))
        
        # Second convolution block (up) - no activation yet
        x = self.bn2(self.conv2(x))
        
        # Apply residual connection and final activation
        x = F.relu(x + residual)
        
        return x

class EncoderBlockResidual(nn.Module):
    """
    Encoder block for U-Net architecture with residual connections.
    
    Each block consists of:
    - Two Conv2D layers with BatchNorm and ReLU
    - Residual connection from input to after the second conv
    - MaxPool for downsampling
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlockResidual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
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
        
        # Store pre-pooled output for encoder-decoder skip connection
        skip = out  
        
        # Apply pooling
        pooled = self.pool(out)
        
        return pooled, skip


class DecoderBlockResidual(nn.Module):
    """
    Decoder block for U-Net architecture with residual connections.
    
    Each block consists of:
    - TransposedConv2D for upsampling
    - Concatenation with skip connection from encoder
    - First Conv2D to reduce channels
    - Second Conv2D with residual connection
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlockResidual, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        
        # After concatenation, we have in_channels * 2 channels
        self.conv1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Second convolution with residual connection
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
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
        
        return x

class SelfAttention(nn.Module):
    """
    Self-Attention module for feature refinement.
    
    Applies self-attention to the input tensor, allowing the model to focus
    on important regions and capture long-range dependencies.
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Reshape for matrix multiplication
        proj_query = self.query(x).view(batch_size, -1, width*height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width*height)
        
        # Attention map
        attention = F.softmax(torch.bmm(proj_query, proj_key), dim=-1)
        
        proj_value = self.value(x).view(batch_size, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # Residual connection with learnable weight
        return self.gamma * out + x


class DilatedBottleneck(nn.Module):
    """
    Enhanced bottleneck block with dilated convolutions and self-attention.
    
    Uses parallel dilated convolutions with different dilation rates to increase
    the receptive field without increasing the number of parameters, followed by
    self-attention to capture long-range dependencies.
    """
    def __init__(self, in_channels, out_channels):
        super(DilatedBottleneck, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Dilated convolutions with different rates (on out_channels)
        self.dil1 = nn.Conv2d(out_channels, out_channels//4, 3, padding=1, dilation=1)
        self.dil2 = nn.Conv2d(out_channels, out_channels//4, 3, padding=2, dilation=2)
        self.dil3 = nn.Conv2d(out_channels, out_channels//4, 3, padding=4, dilation=4)
        self.dil4 = nn.Conv2d(out_channels, out_channels//4, 3, padding=8, dilation=8)
        
        # Combine dilated outputs and map back to input size for residual connection
        self.bn_dil = nn.BatchNorm2d(out_channels)
        self.conv_out = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.bn_out = nn.BatchNorm2d(in_channels)
        
        # Add self-attention module
        self.attention = SelfAttention(in_channels)
        
    def forward(self, x):
        # Store input for residual connection
        residual = x
        
        # First convolution: in_channels -> out_channels
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Parallel dilated convolutions with different receptive fields
        d1 = F.relu(self.dil1(x))
        d2 = F.relu(self.dil2(x))
        d3 = F.relu(self.dil3(x))
        d4 = F.relu(self.dil4(x))
        
        # Concatenate outputs to reform out_channels
        out = torch.cat([d1, d2, d3, d4], dim=1)
        out = F.relu(self.bn_dil(out))
        
        # Map back to in_channels for residual connection
        out = self.bn_out(self.conv_out(out))
        
        # Apply residual connection
        out = F.relu(out + residual)
        
        # Apply self-attention for global context modeling
        out = self.attention(out)
        
        return out