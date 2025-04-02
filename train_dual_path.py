import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils_general import load_config
from models.unet_residual_dual_path import UNetResidualDualPath
from data.mel_spectrogram_dataset import MelSpectrogramDataModule, H5FileManager

def main():
    """
    Main function for training the enhanced U-Net with Dual-Path processing, Non-Local blocks,
    and SE blocks for mel spectrogram reconstruction with improved background feature capture.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train U-Net with Dual-Path processing for mel spectrogram reconstruction')
    parser.add_argument('--config', type=str, default='config/dual_path_model.yaml',
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
    # Dual-path specific arguments
    parser.add_argument('--disable_dual_path', action='store_true',
                        help='Disable dual-path blocks to compare with baseline')
    parser.add_argument('--dual_path_encoder_layers', type=str, default=None,
                        help='Encoder layers to add dual-path to (comma-separated list, overrides config)')
    parser.add_argument('--dual_path_decoder_layers', type=str, default=None,
                        help='Decoder layers to add dual-path to (comma-separated list, overrides config)')
    # Non-local block specific arguments
    parser.add_argument('--disable_nl', action='store_true',
                        help='Disable non-local blocks')
    
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
    
    # Setup Dual-Path configuration
    if 'dual_path' not in config['model']:
        config['model']['dual_path'] = {}
    
    # Apply Dual-Path arguments
    if args.disable_dual_path:
        config['model']['dual_path']['use_dual_path'] = False
    else:
        config['model']['dual_path']['use_dual_path'] = True
    
    if args.dual_path_encoder_layers is not None:
        # Parse comma-separated list of integers
        dual_path_encoder_layers = [int(x) for x in args.dual_path_encoder_layers.split(',')]
        config['model']['dual_path']['dual_path_encoder_layers'] = dual_path_encoder_layers
    
    if args.dual_path_decoder_layers is not None:
        # Parse comma-separated list of integers
        dual_path_decoder_layers = [int(x) for x in args.dual_path_decoder_layers.split(',')]
        config['model']['dual_path']['dual_path_decoder_layers'] = dual_path_decoder_layers
        
    # Apply Non-Local block arguments
    if args.disable_nl:
        config['model']['nl_blocks']['use_nl_blocks'] = False
    
    # Get save directory from config if not provided
    save_dir = args.save_dir if args.save_dir is not None else config['train']['save_dir']
    save_dir = os.path.join(save_dir, 'unet_residual_dual_path')  # Use a different subfolder for the dual-path model
    
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
            filename='unet_residual_dual_path-best',
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
    
    # Initialize the LightningModule - now using the dual-path enhanced model
    model = UNetResidualDualPath(config)
    
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
    print("Starting U-Net with Dual-Path blocks training...")
    
    # Print Dual-Path configuration
    if config['model']['dual_path'].get('use_dual_path', True):
        print("Dual-Path block configuration:")
        print(f"  Encoder layers: {config['model']['dual_path'].get('dual_path_encoder_layers', [-2, -1])}")
        print(f"  Decoder layers: {config['model']['dual_path'].get('dual_path_decoder_layers', [0, 1])}")
    else:
        print("Dual-Path blocks disabled")
    
    trainer.fit(model, data_module, ckpt_path=args.resume)
    
    # Print information about the best model
    print(f"Training completed. Best model saved at: {best_model_dir}/unet_residual_dual_path-best.ckpt")
    print(f"Best validation loss: {trainer.callback_metrics.get('val_loss', 0):.6f}")

    # Clean up resources
    H5FileManager.get_instance().close_all()
    print("Cleaned up H5 file resources")
    
if __name__ == "__main__":
    main()