import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils_general import load_config
from models.unet import UNet
from data.mel_spectrogram_dataset import MelSpectrogramDataModule

def cleanup():
    """
    Clean up resources when the program is finished.
    """
    from utils.utils_datasets import H5FileManager
    H5FileManager.get_instance().close_all()
    print("Cleaned up H5 file resources")

def main():
    """
    Main function for training the U-Net model using PyTorch Lightning.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train U-Net for mel spectrogram reconstruction using PyTorch Lightning')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
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
    
    # Get save directory from config if not provided
    save_dir = args.save_dir if args.save_dir is not None else config['train']['save_dir']
    
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
        # Save periodic checkpoints based on validation loss
        ModelCheckpoint(
            monitor='val_loss',
            filename='unet-{epoch:02d}-{val_loss:.6f}',
            save_top_k=3,
            mode='min',
            save_last=True,
            dirpath=periodic_checkpoint_dir,
        ),
        # Monitor learning rate
        LearningRateMonitor(logging_interval='epoch'),
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=config['train'].get('lr_patience', 5) * 2,
            mode='min',
            verbose=True
        ),
    ]
    
    # Initialize the LightningModule
    model = UNet(config)
    
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
        precision='32-true',  # Use 32-bit precision
        benchmark=True,      # If reproducibility is not critical, this can speed up training
        enable_checkpointing=True,
    )
    
    # Start training
    print("Starting training with Lightning...")
    trainer.fit(model, data_module, ckpt_path=args.resume)
    
    # Print information about the best model
    print(f"Training completed. Best model saved at: {trainer.checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.6f}")

    # Clean up resources
    cleanup()
    
if __name__ == "__main__":
    main()