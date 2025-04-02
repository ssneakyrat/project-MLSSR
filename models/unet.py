import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import io
from PIL import Image
import matplotlib.pyplot as plt
import torchvision

from models.losses import CombinedLoss
from models.blocks import (
    EncoderBlock, DecoderBlock, DilatedBottleneck, NonLocalBlock,
    EncoderBlockDualPath, DecoderBlockDualPath, LowFrequencyEmphasis
)

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