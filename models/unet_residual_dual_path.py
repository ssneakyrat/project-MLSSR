import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision

from models.losses import CombinedLoss
from models.residual_blocks import EncoderBlockResidual, DecoderBlockResidual, DilatedBottleneck
from models.residual_blocks.se_blocks import EncoderBlockResidualSE, DecoderBlockResidualSE, SqueezeExcitationBlock
from models.residual_blocks.non_local_blocks import NonLocalBlock, NonLocalAttentionBlock
from models.residual_blocks.dual_path_blocks import DualPathBlock, EncoderBlockDualPath, DecoderBlockDualPath
from models.residual_blocks.low_freq_emphasis import LowFrequencyEmphasisModule

class UNetResidualDualPath(pl.LightningModule):
    """
    Enhanced U-Net architecture with residual connections, dual-path processing,
    dilated bottleneck with self-attention, Squeeze-Excitation blocks, and 
    Non-Local blocks for improved mel-spectrogram reconstruction.
    
    Dual-path blocks are added to specified encoder/decoder layers to separately
    process background and foreground features, improving homogeneous background
    feature preservation.
    
    Input: (B, 1, 128, 80) - Batch of mel-spectrograms
    Output: (B, 1, 128, 80) - Reconstructed mel-spectrograms
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Add Low-Frequency Emphasis configuration
        self.lfe_config = config['model'].get('low_freq_emphasis', {
            'use_lfe': True,                   # Enable by default
            'lfe_encoder_layers': 'all',       # Apply to all encoder layers
            'lfe_reduction_ratio': 8           # Default reduction ratio
        })

        # Initialize loss function
        self.loss_alpha = config['model'].get('loss_alpha', 0.8)
        self.loss_beta = config['model'].get('loss_beta', 0.2)
        self.loss_fn = CombinedLoss(alpha=self.loss_alpha, beta=self.loss_beta)
        
        # Get scale factors from config
        self.scale_factor = config['model'].get('scale_factor', 1.0)
        self.layer_count = config['model'].get('layer_count', 4)  # Default to 4 layers
        
        # Non-Local Blocks configuration
        self.nl_blocks_config = config['model'].get('nl_blocks', {
            'use_nl_blocks': True,
            'nl_in_bottleneck': True,
            'nl_mode': 'embedded',  # 'dot', 'gaussian', 'embedded', or 'concatenation'
            'nl_encoder_layers': [-1],  # Indices of encoder layers to add NL blocks (-1 = last)
            'nl_decoder_layers': [0],   # Indices of decoder layers to add NL blocks (0 = first)
            'nl_reduction_ratio': 2,    # Channel reduction ratio for efficiency
            'nl_use_sub_sample': True,  # Whether to use sub-sampling in NL blocks
        })
        
        # Dual-Path Blocks configuration
        self.dual_path_config = config['model'].get('dual_path', {
            'use_dual_path': True,
            'dual_path_encoder_layers': [-2, -1],  # Indices of encoder layers to use dual-path (-1 = last)
            'dual_path_decoder_layers': [0, 1],    # Indices of decoder layers to use dual-path (0 = first)
        })
        
        # Define channel progressions based on layer count
        if self.layer_count == 2:
            encoder_channels = [16, 32]
            bottleneck_channels = 64
            decoder_channels = [16, 1]
        elif self.layer_count == 3:
            encoder_channels = [16, 32, 64]
            bottleneck_channels = 128
            decoder_channels = [32, 16, 1]
        elif self.layer_count == 4:  # Default
            encoder_channels = [16, 32, 64, 128]
            bottleneck_channels = 256
            decoder_channels = [64, 32, 16, 1]
        elif self.layer_count == 5:
            encoder_channels = [16, 32, 64, 128, 256]
            bottleneck_channels = 512
            decoder_channels = [128, 64, 32, 16, 1]
        elif self.layer_count == 6:
            encoder_channels = [16, 32, 64, 128, 256, 512]
            bottleneck_channels = 1024
            decoder_channels = [256, 128, 64, 32, 16, 1]
        else:
            # Generate a progression for other layer counts
            encoder_channels = []
            for i in range(self.layer_count):
                encoder_channels.append(16 * (2**i))
            bottleneck_channels = encoder_channels[-1] * 2
            
            decoder_channels = []
            for i in range(self.layer_count - 1):
                idx = self.layer_count - 2 - i  # Count down from layer_count-2 to 0
                decoder_channels.append(encoder_channels[idx])
            decoder_channels.append(1)  # Output channel
        
        # Get channel dimensions from config, or use the defaults
        config_encoder_channels = config['model'].get('encoder_channels', encoder_channels)
        config_bottleneck_channels = config['model'].get('bottleneck_channels', bottleneck_channels)
        config_decoder_channels = config['model'].get('decoder_channels', decoder_channels)
        
        # Use config provided channels if they match layer count, otherwise use defaults
        if len(config_encoder_channels) == self.layer_count:
            encoder_channels = config_encoder_channels
        else:
            print(f"Warning: config encoder_channels length ({len(config_encoder_channels)}) doesn't match layer_count ({self.layer_count}). Using default progression.")
        
        if len(config_decoder_channels) == self.layer_count:
            decoder_channels = config_decoder_channels
        else:
            print(f"Warning: config decoder_channels length ({len(config_decoder_channels)}) doesn't match layer_count ({self.layer_count}). Using default progression.")
        
        bottleneck_channels = config_bottleneck_channels
        
        # Scale the channel dimensions (except for the input/output channels)
        self.encoder_channels = [1] + [int(c * self.scale_factor) for c in encoder_channels]
        self.bottleneck_channels = int(bottleneck_channels * self.scale_factor)
        self.decoder_channels = [int(c * self.scale_factor) for c in decoder_channels[:-1]] + [1]  # Keep output channel at 1
        
        # Log the scaled architecture
        print(f"UNet Width Scale Factor: {self.scale_factor}")
        print(f"UNet Layer Count: {self.layer_count}")
        print(f"Encoder channels: {self.encoder_channels}")
        print(f"Bottleneck channels: {self.bottleneck_channels}")
        print(f"Decoder channels: {self.decoder_channels}")
        
        # SE reduction ratio - for deeper layers we can use a larger reduction ratio
        self.se_reduction = 16
        
        # Convert negative indices for dual-path layers to positive
        dual_path_encoder_layers = self.dual_path_config.get('dual_path_encoder_layers', [-2, -1])
        dual_path_decoder_layers = self.dual_path_config.get('dual_path_decoder_layers', [0, 1])
        
        for i in range(len(dual_path_encoder_layers)):
            if dual_path_encoder_layers[i] < 0:
                dual_path_encoder_layers[i] = self.layer_count + dual_path_encoder_layers[i]
        
        # Create encoder blocks using ModuleList for dynamic layer count
        # Use SE blocks for deeper layers and dual-path blocks for specified layers
        self.encoder_blocks = nn.ModuleList()
        se_layer_threshold = self.layer_count // 2  # Apply SE to deeper half of layers
        
        for i in range(self.layer_count):
            if self.dual_path_config.get('use_dual_path', True) and i in dual_path_encoder_layers:
                # Use dual-path block
                self.encoder_blocks.append(
                    EncoderBlockDualPath(
                        self.encoder_channels[i], 
                        self.encoder_channels[i+1]
                    )
                )
                print(f"Adding Dual-Path to encoder layer {i+1}")
            elif i >= se_layer_threshold:
                # Use SE block for deeper layers
                self.encoder_blocks.append(
                    EncoderBlockResidualSE(
                        self.encoder_channels[i], 
                        self.encoder_channels[i+1],
                        reduction=self.se_reduction
                    )
                )
                print(f"Adding SE to encoder layer {i+1}")
            else:
                # Use standard residual block
                self.encoder_blocks.append(
                    EncoderBlockResidual(
                        self.encoder_channels[i], 
                        self.encoder_channels[i+1]
                    )
                )
        
        # Bottleneck - Use the DilatedBottleneck with attention
        self.bottleneck = DilatedBottleneck(
            self.encoder_channels[-1], 
            self.bottleneck_channels, 
            config['model'].get('attention_head', 4)
        )
        
        # Add Non-Local Block to the bottleneck if specified
        if self.nl_blocks_config.get('use_nl_blocks', True) and self.nl_blocks_config.get('nl_in_bottleneck', True):
            self.bottleneck_nl = NonLocalBlock(
                in_channels=self.encoder_channels[-1],
                inter_channels=self.encoder_channels[-1] // self.nl_blocks_config.get('nl_reduction_ratio', 2),
                mode=self.nl_blocks_config.get('nl_mode', 'embedded'),
                sub_sample=False  # Disable sub-sampling in bottleneck to avoid dimension issues
            )
            print(f"Added Non-Local Block to bottleneck with mode '{self.nl_blocks_config.get('nl_mode', 'embedded')}'")
        else:
            self.bottleneck_nl = None
        
        # Create Non-Local Blocks for specified encoder layers
        if self.nl_blocks_config.get('use_nl_blocks', True):
            self.encoder_nl_blocks = nn.ModuleList()
            nl_encoder_layers = self.nl_blocks_config.get('nl_encoder_layers', [-1])
            
            # Convert negative indices to positive
            for i in range(len(nl_encoder_layers)):
                if nl_encoder_layers[i] < 0:
                    nl_encoder_layers[i] = self.layer_count + nl_encoder_layers[i]
            
            # Create each Non-Local Block
            for i in range(self.layer_count):
                if i in nl_encoder_layers:
                    self.encoder_nl_blocks.append(
                        NonLocalBlock(
                            in_channels=self.encoder_channels[i+1],
                            inter_channels=self.encoder_channels[i+1] // self.nl_blocks_config.get('nl_reduction_ratio', 2),
                            mode=self.nl_blocks_config.get('nl_mode', 'embedded'),
                            sub_sample=False  # Disable sub-sampling in encoder to avoid dimension issues
                        )
                    )
                    print(f"Added Non-Local Block to encoder layer {i+1}")
                else:
                    self.encoder_nl_blocks.append(None)
        else:
            self.encoder_nl_blocks = None
        
        # Create decoder blocks using ModuleList for dynamic layer count
        # Use SE blocks for deeper layers and dual-path blocks for specified layers
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(self.layer_count):
            if i == 0:
                in_channels = self.encoder_channels[-1]
            else:
                in_channels = self.decoder_channels[i-1]
            
            if self.dual_path_config.get('use_dual_path', True) and i in dual_path_decoder_layers:
                # Use dual-path block
                self.decoder_blocks.append(
                    DecoderBlockDualPath(
                        in_channels, 
                        self.decoder_channels[i]
                    )
                )
                print(f"Adding Dual-Path to decoder layer {i+1}")
            elif i < se_layer_threshold:
                # Use SE block for deeper layers (earlier in decoder)
                self.decoder_blocks.append(
                    DecoderBlockResidualSE(
                        in_channels, 
                        self.decoder_channels[i],
                        reduction=self.se_reduction
                    )
                )
                print(f"Adding SE to decoder layer {i+1}")
            else:
                # Use standard residual block
                self.decoder_blocks.append(
                    DecoderBlockResidual(
                        in_channels, 
                        self.decoder_channels[i]
                    )
                )
        
        # Create Non-Local Blocks for specified decoder layers
        if self.nl_blocks_config.get('use_nl_blocks', True):
            self.decoder_nl_blocks = nn.ModuleList()
            nl_decoder_layers = self.nl_blocks_config.get('nl_decoder_layers', [0])
            
            # Create each Non-Local Block
            for i in range(self.layer_count):
                if i in nl_decoder_layers:
                    self.decoder_nl_blocks.append(
                        NonLocalBlock(
                            in_channels=self.decoder_channels[i],
                            inter_channels=self.decoder_channels[i] // self.nl_blocks_config.get('nl_reduction_ratio', 2),
                            mode=self.nl_blocks_config.get('nl_mode', 'embedded'),
                            sub_sample=False  # Disable sub-sampling in decoder to avoid dimension issues
                        )
                    )
                    print(f"Added Non-Local Block to decoder layer {i+1}")
                else:
                    self.decoder_nl_blocks.append(None)
        else:
            self.decoder_nl_blocks = None
        
        # Create Low-Frequency Emphasis modules for encoder layers
        if self.lfe_config.get('use_lfe', True):
            self.lfe_modules = nn.ModuleList()
            lfe_layers = self.lfe_config.get('lfe_encoder_layers', 'all')
            lfe_reduction_ratio = self.lfe_config.get('lfe_reduction_ratio', 8)
            
            for i in range(self.layer_count):
                if lfe_layers == 'all' or i in (lfe_layers if isinstance(lfe_layers, list) else []):
                    self.lfe_modules.append(
                        LowFrequencyEmphasisModule(
                            self.encoder_channels[i+1],
                            reduction_ratio=lfe_reduction_ratio
                        )
                    )
                    print(f"Adding Low-Frequency Emphasis to encoder layer {i+1}")
                else:
                    self.lfe_modules.append(None)
        else:
            self.lfe_modules = None

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
        # Store skip connections
        skip_connections = []
        
        # Encoder path
        for i, encoder in enumerate(self.encoder_blocks):
            x, skip = encoder(x)
            
            # Store the skip connection BEFORE applying LFE
            # This ensures the decoder receives skip connections with expected dimensions
            skip_connections.append(skip)
            
            # Apply Low-Frequency Emphasis if enabled (to the feature map, not the skip connection)
            # This way we're enhancing the features without breaking the skip connection dimensions
            if self.lfe_config.get('use_lfe', True) and self.lfe_modules is not None:
                if self.lfe_modules[i] is not None:
                    x = self.lfe_modules[i](x)  # Apply to features continuing in encoder path
            
            # Apply Non-Local Block to encoder output if specified
            if self.nl_blocks_config.get('use_nl_blocks', True) and self.encoder_nl_blocks is not None:
                if self.encoder_nl_blocks[i] is not None:
                    x = self.encoder_nl_blocks[i](x)
        
        # Apply Non-Local Block to bottleneck input if specified
        if self.nl_blocks_config.get('use_nl_blocks', True) and self.bottleneck_nl is not None:
            x = self.bottleneck_nl(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path with skip connections
        for i, decoder in enumerate(self.decoder_blocks):
            # Use skip connections in reverse order
            skip = skip_connections[self.layer_count - 1 - i]
            x = decoder(x, skip)
            
            # Apply Non-Local Block to decoder output if specified
            if self.nl_blocks_config.get('use_nl_blocks', True) and self.decoder_nl_blocks is not None:
                if self.decoder_nl_blocks[i] is not None:
                    x = self.decoder_nl_blocks[i](x)
        
        # Final output
        x = self.output(x)
        
        return x

    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch (torch.Tensor): Batch of mel spectrograms [B, 1, T, F]
            batch_idx (int): Batch index
            
        Returns:
            dict: Dictionary with loss and logs
        """
        # Get input and target (same for autoencoder)
        x = batch
        y = batch  # Reconstruction task (same input and target)
        
        # Forward pass
        y_pred = self(x)
        
        # Calculate loss
        loss = self.loss_fn(y_pred, y)
        
        # Log training loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate L1 loss separately for monitoring
        l1_loss = F.l1_loss(y_pred, y)
        self.log('train_l1_loss', l1_loss, on_step=False, on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch (torch.Tensor): Batch of mel spectrograms [B, 1, T, F]
            batch_idx (int): Batch index
            
        Returns:
            dict: Dictionary with loss and logs
        """
        # Get input and target (same for autoencoder)
        x = batch
        y = batch  # Reconstruction task
        
        # Forward pass
        y_pred = self(x)
        
        # Calculate loss
        loss = self.loss_fn(y_pred, y)
        
        # Log validation loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate L1 loss separately for monitoring
        l1_loss = F.l1_loss(y_pred, y)
        self.log('val_l1_loss', l1_loss, on_step=False, on_epoch=True, logger=True)
        
        # Log validation images if it's the first batch and the current epoch is a multiple of val_every_epoch
        val_every_epoch = self.config['validation'].get('val_every_epoch', 1)
        if batch_idx == 0 and self.current_epoch % val_every_epoch == 0:
            self._log_validation_images(x, y_pred)
        
        return loss
    
    def _log_validation_images(self, inputs, predictions):
        """
        Log validation images to TensorBoard.
        
        Args:
            inputs (torch.Tensor): Input mel spectrograms [B, 1, T, F]
            predictions (torch.Tensor): Predicted mel spectrograms [B, 1, T, F]
        """
        # Select a few samples to visualize using max_samples from config
        max_samples = self.config['validation'].get('max_samples', 5)
        batch_size = inputs.size(0)
        num_samples = min(max_samples, batch_size)
        
        # Generate random indices to select samples
        indices = torch.randperm(batch_size)[:num_samples]
        
        # Create a figure for each sample
        for i, idx in enumerate(indices):
            # Get input and prediction using random index
            input_mel = inputs[idx, 0].cpu().numpy()  # Shape: [T, F]
            pred_mel = predictions[idx, 0].cpu().numpy()  # Shape: [T, F]
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Plot input mel spectrogram
            im1 = ax1.imshow(input_mel, origin='lower', aspect='auto', cmap='viridis')
            ax1.set_title('Input Mel Spectrogram')
            ax1.set_xlabel('Time Frames')
            ax1.set_ylabel('Mel Bins')
            fig.colorbar(im1, ax=ax1)
            
            # Plot predicted mel spectrogram
            im2 = ax2.imshow(pred_mel, origin='lower', aspect='auto', cmap='viridis')
            ax2.set_title('Reconstructed Mel Spectrogram')
            ax2.set_xlabel('Time Frames')
            ax2.set_ylabel('Mel Bins')
            fig.colorbar(im2, ax=ax2)
            
            # Set tight layout
            plt.tight_layout()
            
            # Convert figure to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            img_tensor = torchvision.transforms.ToTensor()(img)
            
            # Log image to TensorBoard
            self.logger.experiment.add_image(
                f'val_sample_{i}', 
                img_tensor, 
                self.current_epoch
            )
            
            # Close figure to free memory
            plt.close(fig)
        
    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            dict: Dictionary with optimizer and scheduler
        """
        # Get optimizer parameters from config
        learning_rate = self.config['train'].get('learning_rate', 0.001)
        weight_decay = self.config['train'].get('weight_decay', 0.0001)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Get scheduler parameters from config
        scheduler_type = self.config['train'].get('lr_scheduler', 'reduce_on_plateau')
        
        # Create scheduler
        if scheduler_type == 'reduce_on_plateau':
            lr_patience = self.config['train'].get('lr_patience', 5)
            lr_factor = self.config['train'].get('lr_factor', 0.5)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=lr_factor,
                patience=lr_patience,
                verbose=True
            )
            
            # Return dictionary with optimizer and scheduler
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        # Step scheduler
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }