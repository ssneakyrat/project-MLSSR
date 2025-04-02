import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import torchvision.transforms as T

from models.unet import UNet
from models.losses import CombinedLoss


class UNetLightning(pl.LightningModule):
    """
    PyTorch Lightning module for U-Net mel-spectrogram reconstruction.
    """
    def __init__(self, config):
        """
        Initialize the Lightning module.
        
        Args:
            config (dict): Configuration dictionary
        """
        super(UNetLightning, self).__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create U-Net model
        self.model = UNet()
        
        # Loss function
        self.criterion = CombinedLoss(
            alpha=config['model'].get('loss_alpha', 0.8),
            beta=config['model'].get('loss_beta', 0.2)
        )
        
        # Metrics
        self.train_loss = 0.0
        self.val_loss = 0.0

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Model output
        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch (Tensor): Input batch
            batch_idx (int): Batch index
            
        Returns:
            dict: Loss and log information
        """
        # Forward pass
        output = self(batch)
        loss = self.criterion(output, batch)
        
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss}
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch (Tensor): Input batch
            batch_idx (int): Batch index
            
        Returns:
            dict: Loss and visualization data
        """
        # Forward pass
        output = self(batch)
        loss = self.criterion(output, batch)
        
        # Log loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Generate visualizations only for the first batch
        if batch_idx == 0:
            # Create visualizations for up to 4 samples in the batch
            val_imgs = []
            for i in range(min(4, batch.size(0))):
                input_mel = batch[i].unsqueeze(0)  # Add batch dimension back
                output_mel = output[i].unsqueeze(0)
                
                # Create comparison image
                input_tensor, output_tensor, comparison_tensor = self.create_comparison_image(
                    input_mel, output_mel
                )
                val_imgs.append((input_tensor, output_tensor, comparison_tensor))
            
            return {'loss': loss, 'val_imgs': val_imgs}
        
        return {'loss': loss}
    
    def validation_epoch_end(self, outputs):
        """
        Process validation epoch end.
        
        Args:
            outputs (list): List of outputs from validation_step
        """
        # Check if we have visualization data
        if outputs and 'val_imgs' in outputs[0]:
            val_imgs = outputs[0]['val_imgs']
            
            # Log sample images
            for i, (input_img, output_img, comparison_img) in enumerate(val_imgs):
                # Log individual images
                self.logger.experiment.add_image(
                    f'Sample {i+1}/Input', 
                    input_img, 
                    self.current_epoch
                )
                self.logger.experiment.add_image(
                    f'Sample {i+1}/Output', 
                    output_img, 
                    self.current_epoch
                )
                self.logger.experiment.add_image(
                    f'Sample {i+1}/Comparison', 
                    comparison_img, 
                    self.current_epoch
                )
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            dict: Optimizer and scheduler configuration
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config['train'].get('learning_rate', 0.001),
            weight_decay=self.config['train'].get('weight_decay', 0.0001)
        )
        
        scheduler_type = self.config['train'].get('lr_scheduler', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config['train'].get('lr_factor', 0.5),
                patience=self.config['train'].get('lr_patience', 5),
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
            # Default to StepLR as fallback
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
    
    def create_comparison_image(self, input_mel, output_mel):
        """
        Create a comparison image showing input and output mel spectrograms side by side.
        
        Args:
            input_mel (torch.Tensor): Input mel spectrogram [1, T, F]
            output_mel (torch.Tensor): Output (reconstructed) mel spectrogram [1, T, F]
            
        Returns:
            tuple: (input_tensor, output_tensor, comparison_tensor) normalized image tensors
                   in the format expected by TensorBoard (C, H, W)
        """
        # Convert to numpy arrays
        input_np = input_mel.squeeze(0).cpu().numpy()
        output_np = output_mel.squeeze(0).cpu().numpy()
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Plot input spectrogram
        im1 = ax1.imshow(input_np, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title('Input Mel-Spectrogram')
        ax1.set_ylabel('Mel Bins')
        ax1.set_xlabel('Time Frames')
        
        # Plot output spectrogram
        im2 = ax2.imshow(output_np, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title('Reconstructed Mel-Spectrogram')
        ax2.set_ylabel('Mel Bins')
        ax2.set_xlabel('Time Frames')
        
        # Add colorbar
        plt.colorbar(im1, ax=ax1, format='%+2.0f dB')
        plt.colorbar(im2, ax=ax2, format='%+2.0f dB')
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert figure to image
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        # Convert to PIL Image and then to tensor
        from PIL import Image
        comparison_img = Image.open(buf)
        to_tensor = T.ToTensor()
        comparison_tensor = to_tensor(comparison_img)
        
        # Create separate input and output tensors for individual TensorBoard logging
        # Normalize each tensor to [0, 1] for TensorBoard display
        input_tensor = torch.from_numpy(input_np).unsqueeze(0)  # Add channel dimension
        input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min() + 1e-8)
        
        output_tensor = torch.from_numpy(output_np).unsqueeze(0)  # Add channel dimension
        output_tensor = (output_tensor - output_tensor.min()) / (output_tensor.max() - output_tensor.min() + 1e-8)
        
        return input_tensor, output_tensor, comparison_tensor