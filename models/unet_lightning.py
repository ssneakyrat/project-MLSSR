import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision

from models.unet import UNet
from models.losses import CombinedLoss
from utils.utils_transform import normalize_mel_spectrogram


class UNetLightning(pl.LightningModule):
    """
    PyTorch Lightning implementation of the U-Net model for mel-spectrogram reconstruction.
    """
    def __init__(self, config):
        """
        Initialize the model.
        
        Args:
            config (dict): Configuration dictionary
        """
        super(UNetLightning, self).__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model architecture
        self.unet = UNet()
        
        # Initialize loss function
        self.loss_alpha = config['model'].get('loss_alpha', 0.8)
        self.loss_beta = config['model'].get('loss_beta', 0.2)
        self.loss_fn = CombinedLoss(alpha=self.loss_alpha, beta=self.loss_beta)
        
        # Track metrics
        self.train_loss = []
        self.val_loss = []
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, 1, T, F]
            
        Returns:
            torch.Tensor: Output tensor of shape [B, 1, T, F]
        """
        return self.unet(x)
    
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
    
    def on_save_checkpoint(self, checkpoint):
        """
        Add custom data to checkpoint.
        
        Args:
            checkpoint (dict): Checkpoint dictionary
        """
        # Add UNet state dict to checkpoint
        checkpoint['unet_state_dict'] = self.unet.state_dict()
        
    def on_load_checkpoint(self, checkpoint):
        """
        Load custom data from checkpoint.
        
        Args:
            checkpoint (dict): Checkpoint dictionary
        """
        # Load UNet state dict from checkpoint if it exists
        if 'unet_state_dict' in checkpoint:
            self.unet.load_state_dict(checkpoint['unet_state_dict'])