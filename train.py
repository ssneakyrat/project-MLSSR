import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils_general import load_config
from models.unet_residual_nl_se import UNetResidualNLSE
from data.mel_spectrogram_dataset import MelSpectrogramDataModule, H5FileManager

def main():
    """
    Main function for training the enhanced U-Net with Non-Local blocks and SE blocks for mel spectrogram reconstruction.
    This model specifically targets better homogeneous background feature reconstruction.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Enhanced U-Net with Non-Local blocks for mel spectrogram reconstruction')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--h5_path', type=str, default=None,
                        help='Path to H5 file (overrides config)')
    parser.add_argument('--data_key', type=str, default=None,
                        help='Key for mel spectrograms in H5 file (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints and logs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--val_split', type=float, default=None,
                        help='Validation split ratio (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    # Add dataset subset arguments
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use from dataset (overrides config)')
    parser.add_argument('--sample_percentage', type=float, default=None,
                        help='Percentage of dataset to use (0.0-1.0, overrides config)')
    # Non-local block specific arguments
    parser.add_argument('--nl_mode', type=str, default=None, choices=['gaussian', 'embedded', 'dot', 'concatenation'],
                        help='Non-local block mode (overrides config)')
    parser.add_argument('--nl_encoder_layers', type=str, default=None,
                        help='Encoder layers to add non-local blocks to (comma-separated list, overrides config)')
    parser.add_argument('--nl_decoder_layers', type=str, default=None,
                        help='Decoder layers to add non-local blocks to (comma-separated list, overrides config)')
    parser.add_argument('--nl_reduction_ratio', type=int, default=None,
                        help='Channel reduction ratio for non-local blocks (overrides config)')
    parser.add_argument('--disable_nl', action='store_true',
                        help='Disable non-local blocks to compare with baseline')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    pl.seed_everything(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments if provided
    if args.batch_size is not None:
        config['train']['batch_size'] = args.batch_size
        
    if args.epochs is not None:
        config['train']['num_epochs'] = args.epochs
        
    if args.val_split is not None:
        config['train']['validation_split'] = args.val_split
    
    if args.h5_path is not None:
        # Override H5 path in config
        # Extract directory and filename
        h5_dir = os.path.dirname(args.h5_path)
        h5_file = os.path.basename(args.h5_path)
        config['data']['bin_dir'] = h5_dir
        config['data']['bin_file'] = h5_file
    
    if args.data_key is not None:
        config['data']['data_key'] = args.data_key
    
    # Override dataset subset settings if provided
    if args.max_samples is not None:
        config['data']['max_samples'] = args.max_samples
    
    if args.sample_percentage is not None:
        config['data']['sample_percentage'] = args.sample_percentage
    
    # Setup Non-Local block configuration
    if 'nl_blocks' not in config['model']:
        config['model']['nl_blocks'] = {}
    
    # Apply Non-Local block arguments
    if args.disable_nl:
        config['model']['nl_blocks']['use_nl_blocks'] = False
    else:
        config['model']['nl_blocks']['use_nl_blocks'] = True
    
    if args.nl_mode is not None:
        config['model']['nl_blocks']['nl_mode'] = args.nl_mode
    
    if args.nl_encoder_layers is not None:
        # Parse comma-separated list of integers
        nl_encoder_layers = [int(x) for x in args.nl_encoder_layers.split(',')]
        config['model']['nl_blocks']['nl_encoder_layers'] = nl_encoder_layers
    
    if args.nl_decoder_layers is not None:
        # Parse comma-separated list of integers
        nl_decoder_layers = [int(x) for x in args.nl_decoder_layers.split(',')]
        config['model']['nl_blocks']['nl_decoder_layers'] = nl_decoder_layers
    
    if args.nl_reduction_ratio is not None:
        config['model']['nl_blocks']['nl_reduction_ratio'] = args.nl_reduction_ratio
    
    # Get save directory from config if not provided
    save_dir = args.save_dir if args.save_dir is not None else config['train']['save_dir']
    save_dir = os.path.join(save_dir, 'unet_residual_nl_se')  # Use a different subfolder for the NL+SE model
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name='lightning_logs'
    )
    
    # Create subdirectories for different types of checkpoints
    periodic_checkpoint_dir = os.path.join(save_dir, 'periodic_checkpoints')
    best_model_dir = os.path.join(save_dir, 'best_model')
    os.makedirs(periodic_checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # Define callbacks with separate directories for each ModelCheckpoint
    callbacks = [
        # Save best model checkpoint
        ModelCheckpoint(
            monitor='val_loss',
            filename='unet_residual_nl_se-best',
            save_top_k=1,
            mode='min',
            dirpath=best_model_dir,
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval='epoch'),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=50,  # More patience for this enhanced model
            mode='min',
            verbose=False
        ),
    ]
    
    # Initialize the LightningModule - now using the enhanced model with Non-Local blocks
    model = UNetResidualNLSE(config)
    
    # Initialize the DataModule
    data_module = MelSpectrogramDataModule(config)
    
    # Initialize the Trainer
    trainer = pl.Trainer(
        max_epochs=config['train']['num_epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['train'].get('log_interval', 10),
        deterministic=True,  # Ensure reproducibility
        accelerator='auto',  # Automatically choose GPU if available
        devices='auto',      # Use all available devices
        precision='32-true', # Use 32-bit precision
        benchmark=True,      # If reproducibility is not critical, this can speed up training
        enable_checkpointing=True,
    )
    
    # Start training
    print("Starting U-Net with Non-Local blocks and SE blocks training...")
    
    # Print Non-Local block configuration
    if config['model']['nl_blocks'].get('use_nl_blocks', True):
        print("Non-Local block configuration:")
        print(f"  Mode: {config['model']['nl_blocks'].get('nl_mode', 'embedded')}")
        print(f"  Encoder layers: {config['model']['nl_blocks'].get('nl_encoder_layers', [-1])}")
        print(f"  Decoder layers: {config['model']['nl_blocks'].get('nl_decoder_layers', [0])}")
        print(f"  Reduction ratio: {config['model']['nl_blocks'].get('nl_reduction_ratio', 2)}")
        print(f"  Using in bottleneck: {config['model']['nl_blocks'].get('nl_in_bottleneck', True)}")
    else:
        print("Non-Local blocks disabled")
    
    trainer.fit(model, data_module, ckpt_path=args.resume)
    
    # Print information about the best model
    print(f"Training completed. Best model saved at: {best_model_dir}/unet_residual_nl_se-best.ckpt")
    print(f"Best validation loss: {trainer.callback_metrics.get('val_loss', 0):.6f}")

    # Clean up resources
    H5FileManager.get_instance().close_all()
    print("Cleaned up H5 file resources")
    
if __name__ == "__main__":
    main()