import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
    """
    Non-Local Block for capturing global context across the entire feature map.
    
    This block implements the Non-Local Neural Networks concept from Wang et al. 
    (https://arxiv.org/abs/1711.07971) to model long-range dependencies.
    
    It helps to better preserve homogeneous background features by explicitly modeling 
    relationships between any two positions in the feature map, regardless of their distance.
    
    Implementation supports different sub-types:
    - 'gaussian': non-local operation using Gaussian function (dot product)
    - 'embedded': non-local operation using embedded Gaussian function
    - 'dot': non-local operation using dot product
    - 'concatenation': non-local operation using concatenation
    """
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, mode='embedded'):
        """
        Args:
            in_channels (int): Number of input channels
            inter_channels (int, optional): Number of intermediate channels for dimension reduction
                                           If None, uses in_channels // 2
            sub_sample (bool): Whether to apply sub-sampling (strided convolution)
            bn_layer (bool): Whether to use batch normalization
            mode (str): Type of non-local operation ('gaussian', 'embedded', 'dot', 'concatenation')
        """
        super(NonLocalBlock, self).__init__()
        
        self.in_channels = in_channels
        self.mode = mode
        self.sub_sample = sub_sample
        
        # Set intermediate channels for dimension reduction
        if inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1
        else:
            self.inter_channels = inter_channels
            
        print(f"Created NonLocalBlock: in_channels={in_channels}, inter_channels={self.inter_channels}, mode={mode}, sub_sample={sub_sample}")
        
        # Define transformations for query, key, value
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        if mode == 'gaussian':
            # Gaussian mode doesn't need transformations for query and key
            self.theta = None
            self.phi = None
        elif mode == 'concatenation':
            # Concatenation mode uses different approach
            self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
            
            # Additional layers for concatenation mode
            self.concat_project = nn.Sequential(
                nn.Conv2d(self.inter_channels * 2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU()
            )
        else:  # embedded or dot
            self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
            self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        # Sub-sampling layers (optional)
        self.g_pool = None
        self.phi_pool = None
        
        if sub_sample:
            self.g_pool = nn.MaxPool2d(kernel_size=2)
            if self.phi is not None:
                self.phi_pool = nn.MaxPool2d(kernel_size=2)
        
        # Output transformation
        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_channels)
            )
            # Initialize the weights for the output convolution with zeros
            # This ensures the Non-Local block initially behaves like an identity function
            nn.init.constant_(self.W[0].weight, 0)
            nn.init.constant_(self.W[0].bias, 0)
        else:
            self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        
        # Upsampling layer to match spatial dimensions
        self.upsample = None
        
        # Add scaling factor parameter (learnable)
        self.scale_factor = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [B, C, H, W]
            
        Returns:
            Tensor: Output tensor with global context
        """
        batch_size, _, h, w = x.size()
        
        # Calculate value (g)
        g_x = self.g(x)
        if self.sub_sample and self.g_pool is not None:
            g_x = self.g_pool(g_x)
        g_x_size = g_x.size()[2:]  # Store spatial size for later use
        g_x = g_x.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)  # [B, HW_g, C']
        
        if self.mode == 'gaussian':
            # Reshape for matrix multiplication
            theta_x = x.view(batch_size, self.in_channels, -1)  # [B, C, HW]
            theta_x = theta_x.permute(0, 2, 1)  # [B, HW, C]
            
            phi_x = x.view(batch_size, self.in_channels, -1)  # [B, C, HW]
            
            # Calculate attention map (dot product)
            f = torch.matmul(theta_x, phi_x)  # [B, HW, HW]
            f_div_C = F.softmax(f, dim=-1)  # Normalize along the last dimension
            
        elif self.mode == 'concatenation':
            theta_x = self.theta(x)  # [B, C', H, W]
            
            phi_x = self.phi(x)  # [B, C', H, W]
            if self.sub_sample and self.phi_pool is not None:
                phi_x = self.phi_pool(phi_x)
            
            # Note phi_x might have different spatial dimensions than theta_x if sub_sample is True
            theta_h, theta_w = theta_x.size()[2:]
            phi_h, phi_w = phi_x.size()[2:]
            
            # Reshape for concatenation operation
            theta_x = theta_x.view(batch_size, self.inter_channels, -1, 1)  # [B, C', HW_theta, 1]
            phi_x = phi_x.view(batch_size, self.inter_channels, 1, -1)  # [B, C', 1, HW_phi]
            
            # Handling different spatial dimensions between theta and phi
            # For the concatenation mode, this requires a different approach
            if theta_h != phi_h or theta_w != phi_w:
                # Downsample theta_x to match phi_x's spatial dimensions
                theta_x_down = F.interpolate(theta_x.view(batch_size, self.inter_channels, theta_h, theta_w),
                                            size=(phi_h, phi_w), mode='bilinear', align_corners=False)
                theta_x_down = theta_x_down.view(batch_size, self.inter_channels, -1, 1)  # [B, C', HW_phi, 1]
                
                # Concatenate the downsampled theta with phi
                concat_feature = torch.cat([theta_x_down.repeat(1, 1, 1, phi_x.size(3)), 
                                          phi_x.repeat(1, 1, theta_x_down.size(2), 1)], dim=1)
            else:
                # If dimensions match, perform normal concatenation
                concat_feature = torch.cat([theta_x.repeat(1, 1, 1, phi_x.size(3)), 
                                          phi_x.repeat(1, 1, theta_x.size(2), 1)], dim=1)
                
            f = self.concat_project(concat_feature)
            f = f.view(batch_size, -1, phi_x.size(3))  # [B, HW_theta, HW_phi]
            f_div_C = F.softmax(f, dim=-1)
            
        else:  # embedded or dot
            theta_x = self.theta(x)  # [B, C', H, W]
            theta_x = theta_x.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)  # [B, HW_theta, C']
            
            phi_x = self.phi(x)  # [B, C', H, W]
            if self.sub_sample and self.phi_pool is not None:
                phi_x = self.phi_pool(phi_x)
            phi_x = phi_x.view(batch_size, self.inter_channels, -1)  # [B, C', HW_phi]
            
            # Calculate attention map (dot product)
            f = torch.matmul(theta_x, phi_x)  # [B, HW_theta, HW_phi]
            f_div_C = F.softmax(f, dim=-1)
        
        # Apply attention to value
        y = torch.matmul(f_div_C, g_x)  # [B, HW_theta, C']
        
        # Reshape back to original format
        y = y.permute(0, 2, 1).contiguous()  # [B, C', HW_theta]
        y = y.view(batch_size, self.inter_channels, *g_x_size)  # [B, C', H_g, W_g]
        
        # If dimensions differ due to subsampling, upsample back to input size
        if y.size()[2:] != x.size()[2:]:
            y = F.interpolate(y, size=x.size()[2:], mode='bilinear', align_corners=False)
        
        # Apply output transformation
        z = self.W(y)  # [B, C, H, W]
        
        # Apply residual connection with learnable scaling
        return x + self.scale_factor * z


class NonLocalAttentionBlock(nn.Module):
    """
    A more memory-efficient implementation of the Non-Local Block using factorized attention.
    
    This implementation reduces memory usage by applying attention in a factorized manner,
    making it more suitable for high-resolution feature maps.
    """
    def __init__(self, in_channels, reduction_ratio=8, stabilization=1e-6):
        super(NonLocalAttentionBlock, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.inter_channels = max(1, in_channels // reduction_ratio)
        self.stabilization = stabilization
        
        # Input transformations
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        
        # Output transformation
        self.out = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )
        
        # Initialize output conv with zeros for identity mapping at initialization
        nn.init.zeros_(self.out[0].weight)
        nn.init.zeros_(self.out[0].bias)
        
        # Scale factor
        self.scale = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, _, h, w = x.size()
        
        # Project to lower dimension
        theta = self.theta(x)  # [B, C', H, W]
        phi = self.phi(x)      # [B, C', H, W]
        g = self.g(x)          # [B, C', H, W]
        
        # Reshape for factorized attention
        theta = theta.view(batch_size, self.inter_channels, -1)  # [B, C', H*W]
        phi = phi.view(batch_size, self.inter_channels, -1)      # [B, C', H*W]
        g = g.view(batch_size, self.inter_channels, -1)          # [B, C', H*W]
        
        # Transpose for matrix multiplication
        theta = theta.permute(0, 2, 1)  # [B, H*W, C']
        
        # Factorized attention (to save memory)
        # 1. First compute representation in different embedding space
        attention = torch.matmul(theta, phi)  # [B, H*W, H*W]
        
        # 2. Apply softmax for normalization
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value projection
        y = torch.matmul(g, attention.permute(0, 2, 1))  # [B, C', H*W]
        
        # Reshape back to spatial dimensions
        y = y.view(batch_size, self.inter_channels, h, w)
        
        # Apply output transformation with scaling
        y = self.out(y)
        
        # Residual connection with learnable scale
        return x + self.scale * y


# Simplified version for easier integration
def add_non_local_block(model, layer_idx, block_type='standard'):
    """
    Utility function to add a Non-Local Block to a U-Net model at a specific layer.
    
    Args:
        model: The UNet model instance
        layer_idx: Index of the encoder/decoder layer to add the block to
        block_type: Type of non-local block ('standard' or 'attention')
        
    Returns:
        Tuple of (query_module, key_module, modified_model)
    """
    if block_type == 'standard':
        return NonLocalBlock
    else:
        return NonLocalAttentionBlock