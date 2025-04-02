import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from utils.utils_general import load_config
from models.unet_lightning import UNetLightning
from data.data_module import MelSpectrogramDataModule


def main():
    """
    Validate a trained PyTorch Lightning U-Net model.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Validate trained U-Net model using PyTorch Lightning')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save validation results')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create TensorBoard logger for validation
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='validation_logs'
    )
    
    # Initialize the DataModule
    data_module = MelSpectrogramDataModule(config)
    data_module.setup()
    
    # Initialize the model from checkpoint
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = UNetLightning.load_from_checkpoint(args.checkpoint, config=config)
    model.eval()
    
    # Create a Lightning Trainer for validation only
    trainer = pl.Trainer(
        logger=logger,
        accelerator='auto',
        devices='auto',
        precision='32-true',
    )
    
    # Validate the model
    print("Running validation...")
    results = trainer.validate(model, datamodule=data_module)
    print(f"Validation loss: {results[0]['val_loss']:.6f}")
    
    # Generate and save sample visualizations
    print(f"Generating {args.num_samples} sample visualizations...")
    val_loader = data_module.val_dataloader()
    
    # Create a directory for sample images
    samples_dir = os.path.join(args.output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Get a batch of data
    batch = next(iter(val_loader))
    batch = batch.to(model.device)
    
    # Generate reconstructions
    with torch.no_grad():
        output = model(batch)
    
    # Create and save visualizations
    for i in range(min(args.num_samples, batch.size(0))):
        input_mel = batch[i].squeeze(0).cpu().numpy()
        output_mel = output[i].squeeze(0).cpu().numpy()
        
        # Create a figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot input spectrogram
        im1 = ax1.imshow(input_mel, aspect='auto', origin='lower', cmap='viridis')
        ax1.set_title('Input Mel-Spectrogram')
        ax1.set_ylabel('Mel Bins')
        ax1.set_xlabel('Time Frames')
        
        # Plot output spectrogram
        im2 = ax2.imshow(output_mel, aspect='auto', origin='lower', cmap='viridis')
        ax2.set_title('Reconstructed Mel-Spectrogram')
        ax2.set_ylabel('Mel Bins')
        ax2.set_xlabel('Time Frames')
        
        # Add colorbar
        plt.colorbar(im1, ax=ax1, format='%+2.0f dB')
        plt.colorbar(im2, ax=ax2, format='%+2.0f dB')
        
        # Calculate reconstruction error
        mse = np.mean((input_mel - output_mel) ** 2)
        mae = np.mean(np.abs(input_mel - output_mel))
        
        # Add error metrics as a subtitle
        plt.suptitle(f'Sample {i+1} | MSE: {mse:.6f}, MAE: {mae:.6f}')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(samples_dir, f'sample_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved visualization for sample {i+1} to {save_path}")
    
    print(f"Validation completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()