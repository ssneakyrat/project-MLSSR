import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import io
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import numpy as np

from models.blocks import (
    EncoderBlock, DecoderBlock, DilatedBottleneck, NonLocalBlock,
    EncoderBlockDualPath, DecoderBlockDualPath, LowFrequencyEmphasis
)
from models.losses import CombinedLoss

class UNetBase(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        self.loss_alpha = config['model'].get('loss_alpha', 0.8)
        self.loss_beta = config['model'].get('loss_beta', 0.2)
        self.loss_fn = CombinedLoss(alpha=self.loss_alpha, beta=self.loss_beta)
        
        self.scale_factor = config['model'].get('scale_factor', 1.0)
        self.layer_count = config['model'].get('layer_count', 4)
        
        # Set up channel dimensions
        self._setup_channels()
        
        # Initialize model blocks
        self._setup_model()
        
        # Initialize weights
        self._init_weights()
        
    def _setup_channels(self):
        # Define channel progression based on layer count
        if self.layer_count <= 6:
            encoder_channels = []
            for i in range(self.layer_count):
                encoder_channels.append(16 * (2**i))
            bottleneck_channels = encoder_channels[-1] * 2
            
            decoder_channels = []
            for i in range(self.layer_count - 1):
                idx = self.layer_count - 2 - i
                decoder_channels.append(encoder_channels[idx])
            decoder_channels.append(1)
        else:
            # Generate progression for very deep networks
            encoder_channels = []
            for i in range(self.layer_count):
                encoder_channels.append(16 * (2**min(i, 6)))
            bottleneck_channels = encoder_channels[-1] * 2
            
            decoder_channels = []
            for i in range(self.layer_count - 1):
                idx = self.layer_count - 2 - i
                decoder_channels.append(encoder_channels[idx])
            decoder_channels.append(1)
        
        # Apply scale factor
        self.encoder_channels = [1] + [int(c * self.scale_factor) for c in encoder_channels]
        self.bottleneck_channels = int(bottleneck_channels * self.scale_factor)
        self.decoder_channels = [int(c * self.scale_factor) for c in decoder_channels[:-1]] + [1]
        
    def _setup_model(self):
        # Implement in subclasses
        raise NotImplementedError
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Implement in subclasses
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        x = batch
        y = batch
        
        y_pred = self(x)
        
        loss = self.loss_fn(y_pred, y)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_l1_loss', F.l1_loss(y_pred, y), on_step=False, on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        y = batch
        
        y_pred = self(x)
        
        loss = self.loss_fn(y_pred, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_l1_loss', F.l1_loss(y_pred, y), on_step=False, on_epoch=True, logger=True)
        
        # Log validation images
        val_every_epoch = self.config['validation'].get('val_every_epoch', 1)
        if batch_idx == 0 and self.current_epoch % val_every_epoch == 0:
            self._log_validation_images(x, y_pred)
        
        return loss
    
    def _log_validation_images(self, inputs, predictions):
        max_samples = min(self.config['validation'].get('max_samples', 4), inputs.size(0))
        indices = torch.randperm(inputs.size(0))[:max_samples]
        
        for i, idx in enumerate(indices):
            input_mel = inputs[idx, 0].cpu().numpy()
            pred_mel = predictions[idx, 0].cpu().numpy()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            im1 = ax1.imshow(input_mel, origin='lower', aspect='auto', cmap='viridis')
            ax1.set_title('Input Mel Spectrogram')
            ax1.set_xlabel('Time Frames')
            ax1.set_ylabel('Mel Bins')
            fig.colorbar(im1, ax=ax1)
            
            im2 = ax2.imshow(pred_mel, origin='lower', aspect='auto', cmap='viridis')
            ax2.set_title('Reconstructed Mel Spectrogram')
            ax2.set_xlabel('Time Frames')
            ax2.set_ylabel('Mel Bins')
            fig.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            img_tensor = torchvision.transforms.ToTensor()(img)
            
            self.logger.experiment.add_image(f'val_sample_{i}', img_tensor, self.current_epoch)
            
            plt.close(fig)
    
    def configure_optimizers(self):
        learning_rate = self.config['train'].get('learning_rate', 0.001)
        weight_decay = self.config['train'].get('weight_decay', 0.0001)
        
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        scheduler_type = self.config['train'].get('lr_scheduler', 'reduce_on_plateau')
        
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
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
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
        
class UNetResidualDualPath(UNetBase):
    def __init__(self, config):
        super().__init__(config)
        
    def _setup_model(self):
        # Get configurations
        self.nl_blocks_config = self.config['model'].get('nl_blocks', {
            'use_nl_blocks': True,
            'nl_in_bottleneck': True,
            'nl_mode': 'embedded',
            'nl_encoder_layers': [-1],
            'nl_decoder_layers': [0],
            'nl_reduction_ratio': 2,
        })
        
        self.dual_path_config = self.config['model'].get('dual_path', {
            'use_dual_path': True,
            'dual_path_encoder_layers': [-2, -1],
            'dual_path_decoder_layers': [0, 1],
        })
        
        self.lfe_config = self.config['model'].get('low_freq_emphasis', {
            'use_lfe': True,
            'lfe_encoder_layers': 'all',
            'lfe_reduction_ratio': 8
        })
        
        # Convert negative indices for dual-path layers
        dual_path_encoder_layers = self.dual_path_config.get('dual_path_encoder_layers', [-2, -1])
        dual_path_decoder_layers = self.dual_path_config.get('dual_path_decoder_layers', [0, 1])
        
        for i in range(len(dual_path_encoder_layers)):
            if dual_path_encoder_layers[i] < 0:
                dual_path_encoder_layers[i] = self.layer_count + dual_path_encoder_layers[i]
        
        # Create encoder blocks
        self.encoder_blocks = nn.ModuleList()
        se_layer_threshold = self.layer_count // 2
        
        for i in range(self.layer_count):
            if self.dual_path_config.get('use_dual_path', True) and i in dual_path_encoder_layers:
                self.encoder_blocks.append(
                    EncoderBlockDualPath(
                        self.encoder_channels[i], 
                        self.encoder_channels[i+1]
                    )
                )
            elif i >= se_layer_threshold:
                self.encoder_blocks.append(
                    EncoderBlock(
                        self.encoder_channels[i], 
                        self.encoder_channels[i+1],
                        use_se=True,
                        se_reduction=16
                    )
                )
            else:
                self.encoder_blocks.append(
                    EncoderBlock(
                        self.encoder_channels[i], 
                        self.encoder_channels[i+1]
                    )
                )
        
        # Bottleneck
        self.bottleneck = DilatedBottleneck(
            self.encoder_channels[-1], 
            self.bottleneck_channels, 
            self.config['model'].get('attention_head', 4)
        )
        
        # Non-Local Block for bottleneck
        if self.nl_blocks_config.get('use_nl_blocks', True) and self.nl_blocks_config.get('nl_in_bottleneck', True):
            self.bottleneck_nl = NonLocalBlock(
                in_channels=self.encoder_channels[-1],
                inter_channels=self.encoder_channels[-1] // self.nl_blocks_config.get('nl_reduction_ratio', 2),
                mode=self.nl_blocks_config.get('nl_mode', 'embedded')
            )
        else:
            self.bottleneck_nl = None
        
        # Non-Local Blocks for encoder layers
        if self.nl_blocks_config.get('use_nl_blocks', True):
            self.encoder_nl_blocks = nn.ModuleList()
            nl_encoder_layers = self.nl_blocks_config.get('nl_encoder_layers', [-1])
            
            # Convert negative indices to positive
            for i in range(len(nl_encoder_layers)):
                if nl_encoder_layers[i] < 0:
                    nl_encoder_layers[i] = self.layer_count + nl_encoder_layers[i]
            
            for i in range(self.layer_count):
                if i in nl_encoder_layers:
                    self.encoder_nl_blocks.append(
                        NonLocalBlock(
                            in_channels=self.encoder_channels[i+1],
                            inter_channels=self.encoder_channels[i+1] // self.nl_blocks_config.get('nl_reduction_ratio', 2),
                            mode=self.nl_blocks_config.get('nl_mode', 'embedded')
                        )
                    )
                else:
                    self.encoder_nl_blocks.append(None)
        else:
            self.encoder_nl_blocks = None
        
        # Create decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(self.layer_count):
            if i == 0:
                in_channels = self.encoder_channels[-1]
            else:
                in_channels = self.decoder_channels[i-1]
            
            if self.dual_path_config.get('use_dual_path', True) and i in dual_path_decoder_layers:
                self.decoder_blocks.append(
                    DecoderBlockDualPath(
                        in_channels, 
                        self.decoder_channels[i]
                    )
                )
            elif i < se_layer_threshold:
                self.decoder_blocks.append(
                    DecoderBlock(
                        in_channels, 
                        self.decoder_channels[i],
                        use_se=True,
                        se_reduction=16
                    )
                )
            else:
                self.decoder_blocks.append(
                    DecoderBlock(
                        in_channels, 
                        self.decoder_channels[i]
                    )
                )
        
        # Non-Local Blocks for decoder layers
        if self.nl_blocks_config.get('use_nl_blocks', True):
            self.decoder_nl_blocks = nn.ModuleList()
            nl_decoder_layers = self.nl_blocks_config.get('nl_decoder_layers', [0])
            
            for i in range(self.layer_count):
                if i in nl_decoder_layers:
                    self.decoder_nl_blocks.append(
                        NonLocalBlock(
                            in_channels=self.decoder_channels[i],
                            inter_channels=self.decoder_channels[i] // self.nl_blocks_config.get('nl_reduction_ratio', 2),
                            mode=self.nl_blocks_config.get('nl_mode', 'embedded')
                        )
                    )
                else:
                    self.decoder_nl_blocks.append(None)
        else:
            self.decoder_nl_blocks = None
        
        # Low-Frequency Emphasis modules
        if self.lfe_config.get('use_lfe', True):
            self.lfe_modules = nn.ModuleList()
            lfe_layers = self.lfe_config.get('lfe_encoder_layers', 'all')
            lfe_reduction_ratio = self.lfe_config.get('lfe_reduction_ratio', 8)
            
            for i in range(self.layer_count):
                if lfe_layers == 'all' or i in (lfe_layers if isinstance(lfe_layers, list) else []):
                    self.lfe_modules.append(
                        LowFrequencyEmphasis(
                            self.encoder_channels[i+1],
                            reduction_ratio=lfe_reduction_ratio
                        )
                    )
                else:
                    self.lfe_modules.append(None)
        else:
            self.lfe_modules = None
        
        # Final output layer
        self.output = nn.Sigmoid()
    
    def forward(self, x):
        skip_connections = []
        
        # Encoder path
        for i, encoder in enumerate(self.encoder_blocks):
            x, skip = encoder(x)
            skip_connections.append(skip)
            
            # Apply Low-Frequency Emphasis
            if self.lfe_config.get('use_lfe', True) and self.lfe_modules is not None:
                if self.lfe_modules[i] is not None:
                    x = self.lfe_modules[i](x)
            
            # Apply Non-Local Block
            if self.nl_blocks_config.get('use_nl_blocks', True) and self.encoder_nl_blocks is not None:
                if self.encoder_nl_blocks[i] is not None:
                    x = self.encoder_nl_blocks[i](x)
        
        # Apply Non-Local Block to bottleneck
        if self.nl_blocks_config.get('use_nl_blocks', True) and self.bottleneck_nl is not None:
            x = self.bottleneck_nl(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i, decoder in enumerate(self.decoder_blocks):
            skip = skip_connections[self.layer_count - 1 - i]
            x = decoder(x, skip)
            
            # Apply Non-Local Block
            if self.nl_blocks_config.get('use_nl_blocks', True) and self.decoder_nl_blocks is not None:
                if self.decoder_nl_blocks[i] is not None:
                    x = self.decoder_nl_blocks[i](x)
        
        # Final output
        x = self.output(x)
        
        return x
    
class FrequencyBandSplitter(nn.Module):
    """Split and merge frequency bands from mel spectrograms"""
    def __init__(self, mel_bins, num_bands=4, overlap_ratio=0.1):
        super().__init__()
        self.mel_bins = mel_bins
        self.num_bands = num_bands
        self.overlap_ratio = overlap_ratio
        
        # Calculate band boundaries with overlap
        self.band_boundaries = self._calculate_band_boundaries()
        
        # Learnable band weights for recombination
        self.band_weights = nn.Parameter(torch.ones(num_bands))
        
    def _calculate_band_boundaries(self):
        """Calculate band boundaries with logarithmic spacing to match human hearing"""
        # Use logarithmic spacing for more natural frequency bands
        # This gives more resolution to lower frequencies
        log_min = np.log(1)
        log_max = np.log(self.mel_bins)
        log_spacing = np.exp(np.linspace(log_min, log_max, self.num_bands + 1))
        log_spacing = np.round(log_spacing).astype(int)
        
        # Ensure valid ranges and add overlap
        boundaries = []
        overlap = int(self.mel_bins * self.overlap_ratio / self.num_bands)
        
        # Minimum width needed for 2 pooling operations (kernel_size=2)
        min_width = 16  # 2^4 for four max pooling layers
        
        for i in range(self.num_bands):
            start = max(0, log_spacing[i] - overlap)
            end = min(self.mel_bins, log_spacing[i+1] + overlap)
            
            # Ensure minimum width for each band
            if (end - start) < min_width:
                # Calculate how much we need to expand
                expand_by = min_width - (end - start)
                # Expand equally on both sides if possible
                start = max(0, start - expand_by // 2)
                end = min(self.mel_bins, end + expand_by // 2 + expand_by % 2)
                
                # If still not wide enough, expand in available direction
                if (end - start) < min_width:
                    if start > 0:
                        start = max(0, start - (min_width - (end - start)))
                    if (end - start) < min_width and end < self.mel_bins:
                        end = min(self.mel_bins, end + (min_width - (end - start)))
            
            boundaries.append((start, end))
        
        return boundaries
        
    def split(self, x):
        """Split the input spectrogram into frequency bands
        Args:
            x: Input of shape [batch, channels, time, frequency]
        Returns:
            List of tensors, each with a portion of the frequency range
        """
        band_outputs = []
        for start, end in self.band_boundaries:
            # Extract the frequency band
            band = x[:, :, :, start:end]
            band_outputs.append(band)
        return band_outputs
    
    def merge(self, band_outputs, original_shape):
        """Merge processed bands back together
        Args:
            band_outputs: List of tensors, one per frequency band
            original_shape: Target shape for the output
        Returns:
            Merged tensor with shape matching original_shape
        """
        batch, channels, time, freq = original_shape
        output = torch.zeros(original_shape, device=band_outputs[0].device)
        
        # Get normalized weights for band combination
        weights = F.softmax(self.band_weights, dim=0)
        
        # Create overlap masks for smooth transitions
        band_masks = self._create_overlap_masks(original_shape)
        
        # Apply the masks and add contributions
        for i, ((start, end), mask) in enumerate(zip(self.band_boundaries, band_masks)):
            # Pad the band output to match the target slice size
            band_output = band_outputs[i]
            
            # Handle potential dimension mismatch
            if band_output.shape[3] != (end - start):
                band_output = F.interpolate(
                    band_output, 
                    size=(time, end - start),
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Apply band-specific weight
            weighted_output = band_output * weights[i]
            
            # Apply the mask and add to the correct position in the output
            output[:, :, :, start:end] += weighted_output * mask
            
        return output
    
    def _create_overlap_masks(self, original_shape):
        """Create masks for smooth band transitions in overlapping regions"""
        _, _, _, freq = original_shape
        masks = []
        
        for i, (start, end) in enumerate(self.band_boundaries):
            # Create a mask of ones
            mask = torch.ones(end - start, device=self.band_weights.device)
            
            # Add fade-in if not the first band
            if i > 0:
                prev_end = self.band_boundaries[i-1][1]
                if start < prev_end:  # There's an overlap
                    overlap_size = prev_end - start
                    # Linear fade-in over the overlap region
                    fade_in = torch.linspace(0, 1, overlap_size, device=mask.device)
                    mask[:overlap_size] = fade_in
            
            # Add fade-out if not the last band
            if i < len(self.band_boundaries) - 1:
                next_start = self.band_boundaries[i+1][0]
                if end > next_start:  # There's an overlap
                    overlap_size = end - next_start
                    # Linear fade-out over the overlap region
                    fade_out = torch.linspace(1, 0, overlap_size, device=mask.device)
                    mask[-overlap_size:] = fade_out
            
            masks.append(mask)
            
        return masks


class MultiBandUNet(UNetResidualDualPath):
    """U-Net model with separate processing paths for different frequency bands"""
    def __init__(self, config):
        # Initialize with the parent class first
        super().__init__(config)
        
        # Add multi-band specific config
        self.num_bands = config['model'].get('num_freq_bands', 4)
        self.band_overlap = config['model'].get('band_overlap', 0.1)
        
        # Create the band splitter
        self.band_splitter = FrequencyBandSplitter(
            mel_bins=config['model']['mel_bins'],
            num_bands=self.num_bands,
            overlap_ratio=self.band_overlap
        )
        
        # Create separate processing paths for each band
        self._setup_band_specific_paths()
        
        # Create fusion module to combine full-spectrum and band-specific outputs
        self.fusion = nn.Sequential(
            nn.Conv2d(self.num_bands + 1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
    def _setup_band_specific_paths(self):
        """Create specialized processing paths for each frequency band"""
        # Create encoders, bottlenecks, and decoders for each band
        self.band_encoders = nn.ModuleList()
        self.band_bottlenecks = nn.ModuleList()
        self.band_decoders = nn.ModuleList()
        
        # Configuration for band-specific paths
        # We'll make them simpler than the main path
        # IMPORTANT: First channel is 1 (the input has 1 channel), then follow reduced progression
        reduced_encoder_channels = [1] + [max(16, c // 2) for c in self.encoder_channels[1:]]
        reduced_bottleneck_channels = max(32, self.bottleneck_channels // 2)
        reduced_decoder_channels = [max(16, c // 2) for c in self.decoder_channels[:-1]] + [1]
        
        # For each frequency band
        for band_idx in range(self.num_bands):
            # Encoder blocks for this band
            band_encoder = nn.ModuleList()
            for i in range(self.layer_count):
                # For higher bands, use more detailed processing
                use_dual_path = (band_idx >= self.num_bands // 2)
                
                if use_dual_path and i >= self.layer_count - 2:
                    band_encoder.append(
                        EncoderBlockDualPath(
                            reduced_encoder_channels[i],
                            reduced_encoder_channels[i+1]
                        )
                    )
                else:
                    band_encoder.append(
                        EncoderBlock(
                            reduced_encoder_channels[i],
                            reduced_encoder_channels[i+1],
                            use_se=(i >= self.layer_count // 2)
                        )
                    )
            
            # Bottleneck for this band
            band_bottleneck = DilatedBottleneck(
                reduced_encoder_channels[-1],
                reduced_bottleneck_channels,
                num_heads=2  # Fewer heads for efficiency
            )
            
            # Decoder blocks for this band
            band_decoder = nn.ModuleList()
            for i in range(self.layer_count):
                if i == 0:
                    in_channels = reduced_encoder_channels[-1]
                else:
                    in_channels = reduced_decoder_channels[i-1]
                
                if use_dual_path and i <= 1:
                    band_decoder.append(
                        DecoderBlockDualPath(
                            in_channels,
                            reduced_decoder_channels[i]
                        )
                    )
                else:
                    band_decoder.append(
                        DecoderBlock(
                            in_channels,
                            reduced_decoder_channels[i],
                            use_se=(i < self.layer_count // 2)
                        )
                    )
            
            self.band_encoders.append(band_encoder)
            self.band_bottlenecks.append(band_bottleneck)
            self.band_decoders.append(band_decoder)
    
    def forward(self, x):
        """Forward pass with multi-band processing"""
        batch_size, channels, time_frames, freq_bins = x.shape
        
        # 1. Process full spectrogram with the original path
        full_output = super().forward(x)
        
        # 2. Split into frequency bands
        band_inputs = self.band_splitter.split(x)
        band_outputs = []
        
        # 3. Process each band separately with appropriate error handling
        for band_idx, band_input in enumerate(band_inputs):
            # Skip bands that are too small to process
            if band_input.size(2) < 4 or band_input.size(3) < 4:  # Minimum size needed for 2 pooling operations
                # Just create a placeholder of correct shape for this band
                placeholder = torch.zeros_like(band_input)
                band_outputs.append(placeholder)
                continue
                
            skip_connections = []
            
            # Encoder path for this band
            try:
                band_x = band_input
                for i, encoder in enumerate(self.band_encoders[band_idx]):
                    band_x, skip = encoder(band_x)
                    skip_connections.append(skip)
                
                # Bottleneck for this band
                band_x = self.band_bottlenecks[band_idx](band_x)
                
                # Decoder path for this band
                for i, decoder in enumerate(self.band_decoders[band_idx]):
                    skip = skip_connections[self.layer_count - 1 - i]
                    band_x = decoder(band_x, skip)
                
                # Store the processed band
                band_outputs.append(band_x)
                
            except RuntimeError as e:
                # If we encounter any runtime error, use a fallback for this band
                print(f"Error processing band {band_idx}: {e}")
                
                # Just pass through the input as output for this band
                band_outputs.append(band_input)
        
        # 4. Merge band outputs
        band_merged = self.band_splitter.merge(band_outputs, x.shape)
        
        # 5. Combine full spectrum and band-specific processing with the fusion module
        all_outputs = [full_output]
        for band_output in band_outputs:
            # Resize band output to match the full output size
            try:
                resized_band = F.interpolate(
                    band_output, 
                    size=(time_frames, freq_bins),
                    mode='bilinear', 
                    align_corners=False
                )
                all_outputs.append(resized_band)
            except RuntimeError as e:
                print(f"Error resizing band output: {e}")
                # Skip this band if resizing fails
                continue
        
        fused_output = self.fusion(torch.cat(all_outputs, dim=1))
        
        # Store band visualizations for logging
        self.band_visualizations = {
            'inputs': band_inputs,
            'outputs': band_outputs,
            'band_boundaries': self.band_splitter.band_boundaries,
            'merged': band_merged,
            'full': full_output,
            'final': fused_output
        }
        
        return fused_output
    
    def _log_validation_images(self, inputs, predictions):
        """Log validation images including per-band visualizations to TensorBoard"""
        max_samples = min(self.config['validation'].get('max_samples', 4), inputs.size(0))
        indices = torch.randperm(inputs.size(0))[:max_samples]
        
        # Get visualization logging settings from config
        logging_config = self.config['validation'].get('logging', {})
        log_full_spectrum = logging_config.get('full_spectrum', True)
        log_individual_bands = logging_config.get('individual_bands', True)
        log_merged_outputs = logging_config.get('merged_outputs', True)
        log_error_analysis = logging_config.get('error_analysis', True)
        
        for i, idx in enumerate(indices):
            # 1. Log full spectrum (original method) if enabled
            if log_full_spectrum:
                input_mel = inputs[idx, 0].cpu().numpy()
                pred_mel = predictions[idx, 0].cpu().numpy()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                im1 = ax1.imshow(input_mel, origin='lower', aspect='auto', cmap='viridis')
                ax1.set_title('Input Mel Spectrogram')
                ax1.set_xlabel('Time Frames')
                ax1.set_ylabel('Mel Bins')
                fig.colorbar(im1, ax=ax1)
                
                im2 = ax2.imshow(pred_mel, origin='lower', aspect='auto', cmap='viridis')
                ax2.set_title('Reconstructed Mel Spectrogram')
                ax2.set_xlabel('Time Frames')
                ax2.set_ylabel('Mel Bins')
                fig.colorbar(im2, ax=ax2)
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                img_tensor = torchvision.transforms.ToTensor()(img)
                
                self.logger.experiment.add_image(f'val_sample_{i}/full_spectrum', img_tensor, self.current_epoch)
                
                plt.close(fig)
            
            # 2. Log each frequency band if enabled
            if log_individual_bands and hasattr(self, 'band_visualizations'):
                band_inputs = self.band_visualizations['inputs']
                band_outputs = self.band_visualizations['outputs']
                band_boundaries = self.band_visualizations['band_boundaries']
                
                for band_idx, ((start, end), band_input, band_output) in enumerate(
                        zip(band_boundaries, band_inputs, band_outputs)):
                    
                    # Get the band input/output for this sample
                    band_input_mel = band_input[idx, 0].cpu().numpy()
                    band_output_mel = band_output[idx, 0].cpu().numpy()
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    im1 = ax1.imshow(band_input_mel, origin='lower', aspect='auto', cmap='viridis')
                    ax1.set_title(f'Band {band_idx+1} Input (Bins {start}-{end})')
                    ax1.set_xlabel('Time Frames')
                    ax1.set_ylabel('Mel Bins')
                    fig.colorbar(im1, ax=ax1)
                    
                    im2 = ax2.imshow(band_output_mel, origin='lower', aspect='auto', cmap='viridis')
                    ax2.set_title(f'Band {band_idx+1} Output (Bins {start}-{end})')
                    ax2.set_xlabel('Time Frames')
                    ax2.set_ylabel('Mel Bins')
                    fig.colorbar(im2, ax=ax2)
                    
                    plt.tight_layout()
                    
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)
                    img = Image.open(buf)
                    img_tensor = torchvision.transforms.ToTensor()(img)
                    
                    self.logger.experiment.add_image(f'val_sample_{i}/band_{band_idx+1}', img_tensor, self.current_epoch)
                    
                    plt.close(fig)
            
            # 3. Log the band-merged output if enabled
            if log_merged_outputs and hasattr(self, 'band_visualizations'):
                band_merged = self.band_visualizations['merged'][idx, 0].cpu().numpy()
                final_output = self.band_visualizations['final'][idx, 0].cpu().numpy()
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                im1 = ax1.imshow(band_merged, origin='lower', aspect='auto', cmap='viridis')
                ax1.set_title('Band-Merged Output')
                ax1.set_xlabel('Time Frames')
                ax1.set_ylabel('Mel Bins')
                fig.colorbar(im1, ax=ax1)
                
                im2 = ax2.imshow(final_output, origin='lower', aspect='auto', cmap='viridis')
                ax2.set_title('Final Fused Output')
                ax2.set_xlabel('Time Frames')
                ax2.set_ylabel('Mel Bins')
                fig.colorbar(im2, ax=ax2)
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                img_tensor = torchvision.transforms.ToTensor()(img)
                
                self.logger.experiment.add_image(f'val_sample_{i}/merged_outputs', img_tensor, self.current_epoch)
                
                plt.close(fig)
                
            # 4. Create frequency-band analysis visualization if enabled
            if log_error_analysis and hasattr(self, 'band_visualizations'):
                input_mel = inputs[idx, 0].cpu().numpy()
                pred_mel = predictions[idx, 0].cpu().numpy()
                band_boundaries = self.band_visualizations['band_boundaries']
                
                fig, axs = plt.subplots(2, 2, figsize=(14, 10))
                axs = axs.flatten()
                
                # Plot error heatmap (abs difference between input and output)
                error = np.abs(input_mel - pred_mel)
                im = axs[0].imshow(error, origin='lower', aspect='auto', cmap='hot')
                axs[0].set_title('Reconstruction Error Heatmap')
                axs[0].set_xlabel('Time Frames')
                axs[0].set_ylabel('Mel Bins')
                fig.colorbar(im, ax=axs[0])
                
                # Plot average error per frequency bin
                bin_errors = np.mean(error, axis=0)
                axs[1].plot(bin_errors)
                axs[1].set_title('Average Error Per Frequency Bin')
                axs[1].set_xlabel('Mel Bin')
                axs[1].set_ylabel('Average Error')
                
                # Plot band boundaries
                ax = axs[2]
                band_colors = plt.cm.tab10(np.linspace(0, 1, self.num_bands))
                for band_idx, ((start, end), color) in enumerate(zip(band_boundaries, band_colors)):
                    ax.axvspan(start, end, alpha=0.3, color=color, label=f'Band {band_idx+1}')
                
                ax.plot(bin_errors, 'k-', linewidth=2)
                ax.set_title('Band Boundaries and Error Distribution')
                ax.set_xlabel('Mel Bin')
                ax.set_ylabel('Average Error')
                ax.legend()
                
                # Plot spectral analysis of errors
                freq_analysis = np.fft.rfft(bin_errors)
                freq_analysis = np.abs(freq_analysis)
                axs[3].plot(freq_analysis)
                axs[3].set_title('Spectral Analysis of Errors')
                axs[3].set_xlabel('Frequency Component')
                axs[3].set_ylabel('Magnitude')
                
                plt.tight_layout()
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                img_tensor = torchvision.transforms.ToTensor()(img)
                
                self.logger.experiment.add_image(f'val_sample_{i}/error_analysis', img_tensor, self.current_epoch)
                
                plt.close(fig)