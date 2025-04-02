import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResBlock(nn.Module):
    """Residual block with dilated convolutions for Multi-Receptive Field Fusion (MRF)"""
    def __init__(self, channels, kernel_size, dilation):
        super(ResBlock, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size, 
                     dilation=dilation, padding=self._get_padding(kernel_size, dilation)),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=self._get_padding(kernel_size, 1))
        )
        
    def _get_padding(self, kernel_size, dilation):
        return (kernel_size * dilation - dilation) // 2
    
    def forward(self, x):
        return x + self.conv_block(x)


class MRF(nn.Module):
    """Multi-Receptive Field Fusion"""
    def __init__(self, channels, kernel_sizes, dilations):
        super(MRF, self).__init__()
        
        self.resblocks = nn.ModuleList()
        for kernel_size, dilation_rates in zip(kernel_sizes, dilations):
            for dilation in dilation_rates:
                self.resblocks.append(ResBlock(channels, kernel_size, dilation))
    
    def forward(self, x):
        for resblock in self.resblocks:
            x = resblock(x)
        return x


class Generator(nn.Module):
    """HiFi-GAN Generator - Scaled down version"""
    def __init__(self, 
                 in_channels=80,                  # Number of mel bins
                 upsample_initial_channel=128,    # Initial channel count (scaled down from 512)
                 upsample_rates=[8, 8, 2, 2],     # Upsampling factors
                 upsample_kernel_sizes=[16, 16, 4, 4],  # Kernel sizes for upsampling
                 resblock_kernel_sizes=[3, 7, 11],      # Kernel sizes for MRF
                 resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]  # Dilations for MRF
                ):
        super(Generator, self).__init__()
        
        self.in_channels = in_channels
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        
        # Initial projection layer
        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, kernel_size=7, padding=3)
        
        # Upsampling layers
        self.upsamples = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        
        curr_channels = upsample_initial_channel
        for i, (u_rate, u_kernel) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # Calculate output channels (halved at each stage)
            next_channels = curr_channels // 2
            
            # Transposed convolution for upsampling
            self.upsamples.append(
                nn.ConvTranspose1d(
                    curr_channels, next_channels, 
                    kernel_size=u_kernel, 
                    stride=u_rate,
                    padding=(u_kernel - u_rate) // 2
                )
            )
            
            # MRF block after each upsampling
            self.mrfs.append(
                MRF(next_channels, resblock_kernel_sizes, resblock_dilation_sizes)
            )
            
            curr_channels = next_channels
        
        # Final layer
        self.conv_post = nn.Conv1d(curr_channels, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape [B, in_channels, T]
               where T is the number of time frames
        Returns:
            Waveform tensor of shape [B, 1, T*product(upsample_rates)]
        """
        # Initial layer
        x = self.conv_pre(x)
        
        # Upsampling stages
        for upsample, mrf in zip(self.upsamples, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)
            x = mrf(x)
        
        # Final layer and activation
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x


class PeriodDiscriminator(nn.Module):
    """Single period discriminator within MPD"""
    def __init__(self, period, channels=16, max_channels=512):
        super(PeriodDiscriminator, self).__init__()
        
        self.period = period
        norm_f = weight_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, channels, (5, 1), stride=(3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(channels, channels*2, (5, 1), stride=(3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(channels*2, channels*4, (5, 1), stride=(3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(channels*4, channels*8, (5, 1), stride=(3, 1), padding=(2, 0))),
            norm_f(nn.Conv2d(channels*8, min(channels*16, max_channels), (5, 1), 
                            stride=(1, 1), padding=(2, 0))),
        ])
        
        # Final layer
        self.conv_post = norm_f(nn.Conv2d(min(channels*16, max_channels), 1, (3, 1), 
                                         stride=(1, 1), padding=(1, 0)))
    
    def forward(self, x):
        """
        Args:
            x: Input waveform of shape [B, 1, T]
        Returns:
            feature_maps: List of feature maps from intermediate layers
            score: Discrimination score
        """
        batch_size = x.shape[0]
        
        # 2D Reshaping for period-based processing
        if x.size(2) % self.period != 0:
            # Pad if needed
            n_pad = self.period - (x.size(2) % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
        
        # Reshape [B, 1, T] -> [B, 1, T/period, period]
        x = x.view(batch_size, 1, -1, self.period)
        
        feature_maps = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        
        x = self.conv_post(x)
        feature_maps.append(x)
        
        # Flatten output
        x = torch.flatten(x, 1, -1)
        
        return feature_maps, x


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator from HiFi-GAN"""
    def __init__(self, periods=[2, 3, 5, 7, 11], channels=16, max_channels=256):
        super(MultiPeriodDiscriminator, self).__init__()
        
        self.discriminators = nn.ModuleList([
            PeriodDiscriminator(period, channels, max_channels) for period in periods
        ])
    
    def forward(self, x):
        """
        Args:
            x: Input waveform [B, 1, T]
        Returns:
            feature_maps_list: List of feature maps from each discriminator
            scores: List of discrimination scores from each discriminator
        """
        feature_maps_list = []
        scores = []
        
        for disc in self.discriminators:
            feature_maps, score = disc(x)
            feature_maps_list.append(feature_maps)
            scores.append(score)
        
        return feature_maps_list, scores


class ScaleDiscriminator(nn.Module):
    """Single scale discriminator within MSD"""
    def __init__(self, channels=16, max_channels=256, norm_f=nn.utils.weight_norm):
        super(ScaleDiscriminator, self).__init__()
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, channels, 15, stride=1, padding=7)),
            norm_f(nn.Conv1d(channels, channels*2, 41, stride=4, padding=20, groups=4)),
            norm_f(nn.Conv1d(channels*2, channels*4, 41, stride=4, padding=20, groups=16)),
            norm_f(nn.Conv1d(channels*4, channels*8, 41, stride=4, padding=20, groups=16)),
            norm_f(nn.Conv1d(channels*8, min(channels*16, max_channels), 41, stride=4, padding=20, groups=16)),
        ])
        
        self.conv_post = norm_f(nn.Conv1d(min(channels*16, max_channels), 1, 3, stride=1, padding=1))
    
    def forward(self, x):
        """
        Args:
            x: Input waveform of shape [B, 1, T]
        Returns:
            feature_maps: List of feature maps from intermediate layers
            score: Discrimination score
        """
        feature_maps = []
        
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            feature_maps.append(x)
        
        x = self.conv_post(x)
        feature_maps.append(x)
        
        return feature_maps, x


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator from HiFi-GAN"""
    def __init__(self, channels=16, max_channels=256):
        super(MultiScaleDiscriminator, self).__init__()
        
        norm_f = nn.utils.weight_norm
        
        self.discriminators = nn.ModuleList([
            # Original scale
            ScaleDiscriminator(channels, max_channels, norm_f=norm_f),
            # x1/2 downsampled scale
            ScaleDiscriminator(channels, max_channels, norm_f=norm_f),
            # x1/4 downsampled scale
            ScaleDiscriminator(channels, max_channels, norm_f=norm_f)
        ])
        
        self.pooling = nn.AvgPool1d(kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        """
        Args:
            x: Input waveform of shape [B, 1, T]
        Returns:
            feature_maps_list: List of feature maps from each discriminator
            scores: List of discrimination scores from each discriminator
        """
        feature_maps_list = []
        scores = []
        
        # No pooling for the first discriminator (original scale)
        fmaps1, score1 = self.discriminators[0](x)
        feature_maps_list.append(fmaps1)
        scores.append(score1)
        
        # Pool and process with the second discriminator (1/2 scale)
        x_down1 = self.pooling(x)
        fmaps2, score2 = self.discriminators[1](x_down1)
        feature_maps_list.append(fmaps2)
        scores.append(score2)
        
        # Pool again and process with the third discriminator (1/4 scale)
        x_down2 = self.pooling(x_down1)
        fmaps3, score3 = self.discriminators[2](x_down2)
        feature_maps_list.append(fmaps3)
        scores.append(score3)
        
        return feature_maps_list, scores


# Alias for weight_norm to simplify code
weight_norm = nn.utils.weight_norm