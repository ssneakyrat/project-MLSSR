import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import h5py

from utils.utils_general import load_config
from models.unet import UNet
from data.mel_spectrogram_dataset import MelSpectrogramDataModule, H5FileManager

def visualize_samples(h5_path, data_key='mel_spectrograms', num_samples=3, output_dir='results/sample_visualizations'):
    """
    Visualize samples from the H5 file.
    
    Args:
        h5_path (str): Path to the H5 file
        data_key (str): Key for mel spectrograms in the H5 file
        num_samples (int): Number of samples to visualize
        output_dir (str): Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open H5 file
    with h5py.File(h5_path, 'r') as f:
        # Check if data key exists
        if data_key not in f:
            print(f"Error: Data key '{data_key}' not found in {h5_path}")
            return
        
        # Get dataset
        dataset = f[data_key]
        dataset_size = dataset.shape[0]
        
        # Determine number of samples to visualize
        num_to_visualize = min(num_samples, dataset_size)
        
        # Get file IDs if available
        file_ids = None
        if 'file_ids' in f:
            file_ids = f['file_ids']
        
        # Get random indices
        indices = np.random.choice(dataset_size, num_to_visualize, replace=False)
        
        # Visualize samples
        print(f"Visualizing {num_to_visualize} samples from {h5_path}")
        
        for i, idx in enumerate(indices):
            # Get mel spectrogram
            mel_spec = dataset[idx]
            
            # Get file ID if available
            file_id = f"sample_{idx}"
            if file_ids is not None:
                file_id = file_ids[idx]
            
            # Create figure
            plt.figure(figsize=(8, 6))
            plt.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Mel Spectrogram - {file_id}")
            plt.xlabel("Time Frames")
            plt.ylabel("Mel Bins")
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{i+1}_{file_id}.png"), dpi=300)
            plt.close()
            
            print(f"  Saved visualization for {file_id}")

def main():
    """
    Validate a trained U-Net model and visualize dataset samples.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Validate trained U-Net model and dataset')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--h5_path', type=str, default=None,
                        help='Path to H5 file (overrides config)')
    parser.add_argument('--data_key', type=str, default=None,
                        help='Key for mel spectrograms in H5 file (overrides config)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint to validate')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save validation results')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Only visualize dataset samples without model validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments if provided
    if args.h5_path is not None:
        # Extract directory and filename
        h5_dir = os.path.dirname(args.h5_path)
        h5_file = os.path.basename(args.h5_path)
        config['data']['bin_dir'] = h5_dir
        config['data']['bin_file'] = h5_file
    
    if args.data_key is not None:
        config['data']['data_key'] = args.data_key
    
    # Get H5 file path
    h5_path = os.path.join(
        config['data']['bin_dir'],
        config['data']['bin_file']
    )
    
    # Get data key
    data_key = config['data'].get('data_key', 'mel_spectrograms')
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize dataset samples
    sample_viz_dir = os.path.join(args.output_dir, 'dataset_samples')
    visualize_samples(h5_path, data_key, args.num_samples, sample_viz_dir)
    
    # If only visualizing dataset, exit
    if args.visualize_only:
        print("Dataset visualization complete. Exiting as --visualize_only was specified.")
        return
    
    # Check if checkpoint is provided
    if args.checkpoint is None:
        print("No checkpoint provided for model validation. Use --checkpoint to specify a model checkpoint.")
        return
    
    # Create TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name='validation_logs'
    )
    
    # Create directory for model validation results
    model_viz_dir = os.path.join(args.output_dir, 'model_validation')
    os.makedirs(model_viz_dir, exist_ok=True)
    
    # Initialize the DataModule
    data_module = MelSpectrogramDataModule(config)
    data_module.setup()
    
    # Initialize model from checkpoint
    print(f"Loading model from checkpoint: {args.checkpoint}")
    model = UNet.load_from_checkpoint(args.checkpoint, config=config)
    model.eval()
    
    # Create a Lightning Trainer for validation
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
    
    # Generate and save sample reconstructions
    print(f"Generating {args.num_samples} sample reconstructions...")
    val_loader = data_module.val_dataloader()
    
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
        save_path = os.path.join(model_viz_dir, f'reconstruction_{i+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved reconstruction for sample {i+1} to {save_path}")
    
    # Clean up resources
    H5FileManager.get_instance().close_all()
    
    print(f"Validation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()