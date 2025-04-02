import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision

from models.losses import CombinedLoss

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
        self.bn2 = nn.BatchNorm2d(out_channels)
        
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

class UNet(pl.LightningModule):
    """
    U-Net architecture for mel-spectrogram reconstruction.
    
    Input: (B, 1, 128, 80) - Batch of mel-spectrograms
    Output: (B, 1, 128, 80) - Reconstructed mel-spectrograms
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize loss function
        self.loss_alpha = config['model'].get('loss_alpha', 0.8)
        self.loss_beta = config['model'].get('loss_beta', 0.2)
        self.loss_fn = CombinedLoss(alpha=self.loss_alpha, beta=self.loss_beta)
        
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
        
        # Log validation images if it's the first batch of each epoch
        if batch_idx == 0:
            self._log_validation_images(x, y_pred)
        
        return loss
    
    def _log_validation_images(self, inputs, predictions):
        """
        Log validation images to TensorBoard.
        
        Args:
            inputs (torch.Tensor): Input mel spectrograms [B, 1, T, F]
            predictions (torch.Tensor): Predicted mel spectrograms [B, 1, T, F]
        """
        # Select a few samples to visualize
        num_samples = min(4, inputs.size(0))
        
        # Create a figure for each sample
        for i in range(num_samples):
            # Get input and prediction
            input_mel = inputs[i, 0].cpu().numpy()  # Shape: [T, F]
            pred_mel = predictions[i, 0].cpu().numpy()  # Shape: [T, F]
            
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
