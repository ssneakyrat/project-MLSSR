import torch
import torch.nn as nn
import torch.nn.functional as F

class LowFrequencyEmphasisModule(nn.Module):
    """
    Module that emphasizes low-frequency components in mel-spectrograms,
    helping to preserve homogeneous background features.
    
    This implementation uses deterministic operations compatible with 
    torch.use_deterministic_algorithms(True).
    """
    def __init__(self, in_channels, reduction_ratio=8, mel_bins=80):
        super(LowFrequencyEmphasisModule, self).__init__()
        self.in_channels = in_channels
        self.mel_bins = mel_bins
        
        # Instead of adaptive pooling, use regular avg pooling with fixed kernel size
        # This ensures deterministic behavior
        self.freq_pool = nn.AvgPool2d(kernel_size=(1, mel_bins), stride=1)
        
        # Channel-wise processing
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Create learnable frequency bias - higher weights for lower frequencies
        # This is a 2D tensor: [1, mel_bins] that will be expanded later
        freq_bias = torch.ones(1, mel_bins)
        for i in range(mel_bins):
            freq_bias[0, i] = 1.0 - (i / mel_bins) * 0.8  # Higher weight for lower freqs
        
        # Register as parameter so it becomes learnable
        self.freq_bias = nn.Parameter(freq_bias)
    
    def forward(self, x):
        # x shape: [B, C, T, F] - Batch, Channels, Time, Frequency bins
        batch_size, channels, time_steps, freq_bins = x.size()
        
        # If frequency dimension doesn't match expected mel_bins, adjust the bias
        if freq_bins != self.mel_bins:
            # Create a new bias with the correct size
            device = x.device
            new_bias = torch.ones(1, freq_bins, device=device)
            for i in range(freq_bins):
                # Apply the same formula but with correct dimensions
                new_bias[0, i] = 1.0 - (i / freq_bins) * 0.8
            freq_bias = new_bias
        else:
            freq_bias = self.freq_bias
        
        # Global frequency pooling (deterministic)
        # Reshape for pooling if necessary
        if freq_bins != self.mel_bins:
            # Use max pooling with appropriate kernel size
            pooled = F.avg_pool2d(x, kernel_size=(1, freq_bins))
        else:
            pooled = self.freq_pool(x)
        
        # Channel attention
        attn = self.channel_attention(pooled)
        
        # Expand attention to match input frequency dimension
        attn = attn.expand(-1, -1, -1, freq_bins)
        
        # Expand freq_bias to match input dimensions: [1, F] -> [B, C, T, F]
        expanded_bias = freq_bias.unsqueeze(0).unsqueeze(0).expand(batch_size, channels, time_steps, -1)
        
        # Apply both attention and frequency bias
        return x * attn * expanded_bias