import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import load_config
from models.combined_model import MultiBandUNetWithHiFiGAN
from data.dataset_with_audio import MelAudioDataModule, H5FileManager

def main():
    # Enable tensor cores on supported NVIDIA GPUs for better performance
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        # Setting to 'high' for best performance with minimal precision loss
        torch.set_float32_matmul_precision('high')
        print("Enabled high-precision tensor core operations for faster training")
    
    parser = argparse.ArgumentParser(description='Train Combined MultiBand U-Net with HiFi-GAN Vocoder')
    parser.add_argument('--config', type=str, default='config/model_with_vocoder.yaml', help='Path to configuration file')
    parser.add_argument('--h5_path', type=str, default=None, help='Path to H5 file (overrides config)')
    parser.add_argument('--mel_key', type=str, default=None, help='Key for mel spectrograms in H5 file (overrides config)')
    parser.add_argument('--audio_key', type=str, default=None, help='Key for audio waveforms in H5 file (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints and logs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use (overrides config)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    parser.add_argument('--single_process', action='store_true', 
                    help='Disable multiprocessing for data loading (fixes H5py issues)')
    
    # Add arguments for vocoder control
    parser.add_argument('--disable_vocoder', action='store_true', help='Disable vocoder training and use U-Net only')
    parser.add_argument('--vocoder_weights', type=float, default=None, help='Weight for vocoder loss (overrides config)')
    parser.add_argument('--unet_weights', type=float, default=None, help='Weight for U-Net loss (overrides config)')
    parser.add_argument('--disc_start_step', type=int, default=None, help='When to start discriminator training (overrides config)')
    
    # Add argument for tensor core precision
    parser.add_argument('--tc_precision', type=str, choices=['medium', 'high'], default='high',
                        help='Tensor core precision setting for matrix multiplications')
    
    # Add argument for variable length processing
    parser.add_argument('--variable_length', action='store_true', help='Enable variable length processing')
    parser.add_argument('--max_audio_length', type=float, default=None, help='Maximum audio length in seconds')
    
    # Add argument for checkpoint handling
    parser.add_argument('--separate_checkpoints', action='store_true', help='Save separate checkpoints for each component')
    
    args = parser.parse_args()
    
    # Allow command line override of tensor core precision
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        torch.set_float32_matmul_precision(args.tc_precision)
        print(f"Using '{args.tc_precision}' precision for tensor core operations")
    
    pl.seed_everything(args.seed)
    
    config = load_config(args.config)
    
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
    
    if args.mel_key:
        config['data']['data_key'] = args.mel_key
    
    if args.audio_key:
        config['data']['audio_key'] = args.audio_key
    
    if args.max_samples:
        config['data']['max_samples'] = args.max_samples
    
    if args.single_process:
        print("Forcing single process mode for data loading")
        config['train']['num_workers'] = 0

    # Set variable length mode
    if args.variable_length:
        config['data']['variable_length'] = True
        config['model']['variable_length_mode'] = True
    
    # Set maximum audio length
    if args.max_audio_length:
        config['audio']['max_audio_length'] = args.max_audio_length
    
    # Vocoder settings
    if args.disable_vocoder:
        if 'vocoder' not in config:
            config['vocoder'] = {}
        config['vocoder']['enabled'] = False
        print("Vocoder training disabled, running in U-Net only mode")
    elif 'vocoder' not in config:
        # Create vocoder section if it doesn't exist
        config['vocoder'] = {'enabled': True}
        print("Added default vocoder configuration (enabled)")
    
    if args.vocoder_weights:
        if 'vocoder' not in config:
            config['vocoder'] = {}
        config['vocoder']['vocoder_weight'] = args.vocoder_weights
    
    if args.unet_weights:
        if 'vocoder' not in config:
            config['vocoder'] = {}
        config['vocoder']['unet_weight'] = args.unet_weights
    
    if args.disc_start_step:
        if 'vocoder' not in config:
            config['vocoder'] = {}
        config['vocoder']['disc_start_step'] = args.disc_start_step
    
    if args.separate_checkpoints:
        if 'vocoder' not in config:
            config['vocoder'] = {}
        config['vocoder']['separate_checkpointing'] = True
    
    save_dir = args.save_dir or config['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Print configuration summary
    vocoder_enabled = config.get('vocoder', {}).get('enabled', False)
    max_audio_len = config['audio']['max_audio_length']
    sample_rate = config['audio']['sample_rate']
    hop_length = config['audio']['hop_length']
    batch_size = config['train']['batch_size']
    
    frames_per_second = sample_rate / hop_length
    max_frames = int(max_audio_len * frames_per_second)
    
    print(f"Configuration summary:")
    print(f"  - Maximum audio length: {max_audio_len} seconds")
    print(f"  - Maximum mel frames: {max_frames}")
    print(f"  - Maximum audio samples: {int(max_audio_len * sample_rate)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Variable length mode: {config['data'].get('variable_length', False)}")
    print(f"  - Vocoder enabled: {vocoder_enabled}")
    
    if vocoder_enabled:
        print(f"  - Vocoder settings:")
        vocoder_config = config['vocoder']
        print(f"    - U-Net weight: {vocoder_config.get('unet_weight', 1.0)}")
        print(f"    - Vocoder weight: {vocoder_config.get('vocoder_weight', 1.0)}")
        print(f"    - Generator channels: {vocoder_config.get('upsample_initial_channel', 128)}")
        print(f"    - Upsampling rates: {vocoder_config.get('upsample_rates', [8, 8, 2, 2])}")
        print(f"    - Discriminator start step: {vocoder_config.get('disc_start_step', 0)}")
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name='lightning_logs'
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            filename='combined-{epoch:02d}-{val_loss:.4f}',
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
    
    # Initialize model and data module
    model = MultiBandUNetWithHiFiGAN(config)
    data_module = MelAudioDataModule(config)
    
    # Define training configuration
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    # Adjust trainer settings
    use_16bit = config['train'].get('precision', '32-true') == '16-mixed'
    
    # Set up trainer kwargs
    trainer_kwargs = {
        'max_epochs': config['train']['num_epochs'],
        'logger': logger,
        'callbacks': callbacks,
        'log_every_n_steps': config['train'].get('log_interval', 10),
        'deterministic': config.get('train', {}).get('deterministic', False),
        'accelerator': 'auto',
        'devices': 'auto',
        'precision': config['train'].get('precision', '32-true'),
        'benchmark': True,  # Enable cudnn benchmarking for faster training
    }
    
    # Only add gradient accumulation for automatic optimization (U-Net only mode)
    if not vocoder_enabled and 'accumulate_grad_batches' in config['train']:
        trainer_kwargs['accumulate_grad_batches'] = config['train']['accumulate_grad_batches']
        print(f"  - Gradient accumulation: {config['train']['accumulate_grad_batches']} batches")
    elif vocoder_enabled and 'accumulate_grad_batches' in config['train']:
        print(f"  - Note: Automatic gradient accumulation disabled for manual optimization with vocoder")
        print(f"    Original accumulate_grad_batches value: {config['train']['accumulate_grad_batches']}")
        print(f"    Consider adjusting batch size to compensate")
    
    # Create trainer with appropriate settings
    trainer = pl.Trainer(**trainer_kwargs)
    
    try:
        trainer.fit(model, data_module, ckpt_path=args.resume)
        print(f"Training completed. Best model saved with val_loss: {trainer.callback_metrics.get('val_loss', 0):.6f}")
    finally:
        H5FileManager.get_instance().close_all()
    
if __name__ == "__main__":
    main()