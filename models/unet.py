import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderBlock(nn.Module):
    """
    Encoder block for U-Net architecture.
    
    Each block consists of:
    - Two Conv2D layers with BatchNorm and ReLU
    - MaxPool for downsampling
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Store pre-pooled output for skip connection
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        skip = x  # Store for skip connection
        x = self.pool(x)
        return x, skip

class Bottleneck(nn.Module):
    """
    Bottleneck block between encoder and decoder.
    """
    def __init__(self, in_channels, out_channels):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class DecoderBlock(nn.Module):
    """
    Decoder block for U-Net architecture.
    
    Each block consists of:
    - TransposedConv2D for upsampling
    - Concatenation with skip connection
    - Two Conv2D layers with BatchNorm and ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle potential size mismatches
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode='bilinear', align_corners=True)
            
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNet(nn.Module):
    """
    U-Net architecture for mel-spectrogram reconstruction.
    
    Input: (B, 1, 128, 80) - Batch of mel-spectrograms
    Output: (B, 1, 128, 80) - Reconstructed mel-spectrograms
    """
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder path
        self.enc1 = EncoderBlock(1, 16)
        self.enc2 = EncoderBlock(16, 32)
        self.enc3 = EncoderBlock(32, 64)
        self.enc4 = EncoderBlock(64, 128)
        
        # Bottleneck
        self.bottleneck = Bottleneck(128, 256)
        
        # Decoder path
        self.dec1 = DecoderBlock(128, 64)
        self.dec2 = DecoderBlock(64, 32)
        self.dec3 = DecoderBlock(32, 16)
        self.dec4 = DecoderBlock(16, 1)
        
        # Final output layer
        self.output = nn.Sigmoid()  # Ensure output is in [0,1] range
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using He normal initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Encoder path
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)
        
        # Final output
        x = self.output(x)
        
        return x