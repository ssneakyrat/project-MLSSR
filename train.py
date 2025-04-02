import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import load_config
from models.multi_band_unet import MultiBandUNet
from data.dataset import DataModule, H5FileManager

def main():
    # Enable tensor cores on supported NVIDIA GPUs for better performance
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        # Setting to 'high' for best performance with minimal precision loss
        torch.set_float32_matmul_precision('high')
        print("Enabled high-precision tensor core operations for faster training")
    
    parser = argparse.ArgumentParser(description='Train Multi-Band U-Net for mel spectrogram reconstruction')
    parser.add_argument('--config', type=str, default='config/model.yaml', help='Path to configuration file')
    parser.add_argument('--h5_path', type=str, default=None, help='Path to H5 file (overrides config)')
    parser.add_argument('--data_key', type=str, default=None, help='Key for mel spectrograms in H5 file (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints and logs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use (overrides config)')
    parser.add_argument('--num_bands', type=int, default=None, help='Number of frequency bands (overrides config)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Add new arguments for variable length and audio length
    parser.add_argument('--variable_length', action='store_true', help='Enable variable length processing')
    parser.add_argument('--max_audio_length', type=float, default=None, help='Maximum audio length in seconds')
    
    # Add new arguments for visualization toggles
    parser.add_argument('--log_full', type=str, default=None, choices=['true', 'false'], 
                        help='Enable/disable full spectrum logging (overrides config)')
    parser.add_argument('--log_bands', type=str, default=None, choices=['true', 'false'], 
                        help='Enable/disable individual band logging (overrides config)')
    parser.add_argument('--log_merged', type=str, default=None, choices=['true', 'false'], 
                        help='Enable/disable merged outputs logging (overrides config)')
    parser.add_argument('--log_error', type=str, default=None, choices=['true', 'false'], 
                        help='Enable/disable error analysis logging (overrides config)')
    
    # Add argument for tensor core precision
    parser.add_argument('--tc_precision', type=str, choices=['medium', 'high'], default='high',
                        help='Tensor core precision setting for matrix multiplications')
    
    args = parser.parse_args()
    
    # Allow command line override of tensor core precision
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        torch.set_float32_matmul_precision(args.tc_precision)
        print(f"Using '{args.tc_precision}' precision for tensor core operations")
    
    pl.seed_everything(args.seed)
    
    config = load_config(args.config)
    
    # Ensure validation logging section exists
    if 'validation' not in config:
        config['validation'] = {}
    if 'logging' not in config['validation']:
        config['validation']['logging'] = {}
    
    # Apply command line overrides for standard parameters
    if args.batch_size:
        config['train']['batch_size'] = args.batch_size
        
    if args.epochs:
        config['train']['num_epochs'] = args.epochs
    
    if args.h5_path:
        h5_dir = os.path.dirname(args.h5_path)
        h5_file = os.path.basename(args.h5_path)
        config['data']['bin_dir'] = h5_dir
        config['data']['bin_file'] = h5_file
    
    if args.data_key:
        config['data']['data_key'] = args.data_key
    
    if args.max_samples:
        config['data']['max_samples'] = args.max_samples
    
    if args.num_bands:
        config['model']['num_freq_bands'] = args.num_bands
    
    # Set variable length mode
    if args.variable_length:
        config['data']['variable_length'] = True
        config['model']['variable_length_mode'] = True
    
    # Set maximum audio length
    if args.max_audio_length:
        config['audio']['max_audio_length'] = args.max_audio_length
    elif 'max_audio_length' not in config['audio']:
        # Set default 10 second max if not specified
        config['audio']['max_audio_length'] = 10.0
    
    # Apply command line overrides for visualization toggles
    if args.log_full:
        config['validation']['logging']['full_spectrum'] = args.log_full.lower() == 'true'
    
    if args.log_bands:
        config['validation']['logging']['individual_bands'] = args.log_bands.lower() == 'true'
    
    if args.log_merged:
        config['validation']['logging']['merged_outputs'] = args.log_merged.lower() == 'true'
    
    if args.log_error:
        config['validation']['logging']['error_analysis'] = args.log_error.lower() == 'true'
    
    save_dir = args.save_dir or config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Calculate and print the memory requirements
    max_audio_len = config['audio']['max_audio_length']
    sample_rate = config['audio']['sample_rate']
    hop_length = config['audio']['hop_length']
    batch_size = config['train']['batch_size']
    
    frames_per_second = sample_rate / hop_length
    max_frames = int(max_audio_len * frames_per_second)
    
    print(f"Configuration summary:")
    print(f"  - Maximum audio length: {max_audio_len} seconds")
    print(f"  - Maximum time frames: {max_frames}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Variable length mode: {config['data'].get('variable_length', False)}")
    
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name='lightning_logs'
    )
    
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            filename='multiband-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min'
        ),
    ]
    
    model = MultiBandUNet(config)
    data_module = DataModule(config)
    
    # Adjust trainer settings for GPU optimization
    trainer = pl.Trainer(
        max_epochs=config['train']['num_epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['train'].get('log_interval', 10),
        deterministic=config.get('train', {}).get('deterministic', False),  # Set to False for better performance
        accelerator='auto',
        devices='auto',
        precision=config['train'].get('precision', 10),  # Can also use '16-mixed' for further speedup if model allows
        benchmark=True  # Enable cudnn benchmarking for faster training
    )
    
    try:
        trainer.fit(model, data_module, ckpt_path=args.resume)
        print(f"Training completed. Best model saved with val_loss: {trainer.callback_metrics.get('val_loss', 0):.6f}")
    finally:
        H5FileManager.get_instance().close_all()
    
if __name__ == "__main__":
    main()