import os
import torch
import argparse
from data.datasets import load_dataset
from utils.utils_general import load_config
from trainers.trainer import train

def main():
    """
    Main function for training the U-Net model.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train U-Net for mel spectrogram reconstruction')
    parser.add_argument('--config', type=str, default='config/model.yaml',
                        help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default='runs/unet',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--val_split', type=float, default=None,
                        help='Validation split ratio (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments if provided
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
        
    if args.epochs is not None:
        config['train']['num_epochs'] = args.epochs
        
    if args.val_split is not None:
        config['train']['validation_split'] = args.val_split
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    data_items = load_dataset(split='train', shuffle=True)
    
    # Print dataset summary
    print(f"Loaded {len(data_items)} samples")
    
    # Check if at least one sample contains mel spectrogram
    has_mel = any('mel_spec' in item for item in data_items)
    if not has_mel:
        print("Error: No mel spectrograms found in the dataset")
        return
    
    # Start training
    print(f"Starting training with batch size {config['train']['batch_size']}")
    model, val_loss = train(
        config=config,
        data_items=data_items,
        save_dir=args.save_dir,
        num_epochs=config['train']['num_epochs'],
        batch_size=config['train']['batch_size'],
        validation_split=config['train']['validation_split'],
        log_interval=config['train']['log_interval'],
        save_interval=config['train']['save_interval'],
        device=device
    )
    
    print(f"Training completed. Best validation loss: {val_loss:.6f}")

if __name__ == "__main__":
    main()