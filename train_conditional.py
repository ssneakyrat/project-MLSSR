import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import load_config
from models.conditional_multi_band_unet import ConditionalMultiBandUNet
from data.conditional_dataset import ConditionalDataModule, H5FileManager

def main():
    # Enable tensor cores on Ampere+ GPUs for better performance
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7:
        # Setting to 'high' for best performance with minimal precision loss
        torch.set_float32_matmul_precision('high')
        print("Enabled high-precision tensor core operations for faster training")
    
    parser = argparse.ArgumentParser(description='Train Conditional Multi-Band U-Net for mel spectrogram generation')
    parser.add_argument('--config', type=str, default='config/conditional_model.yaml', help='Path to configuration file')
    parser.add_argument('--h5_path', type=str, default=None, help='Path to H5 file (overrides config)')
    parser.add_argument('--data_key', type=str, default=None, help='Key for mel spectrograms in H5 file (overrides config)')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save checkpoints and logs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use (overrides config)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Add arguments for noise conditioning approach
    parser.add_argument('--noise_ramp_steps', type=int, default=None, 
                        help='Number of steps to ramp up noise level (overrides config)')
    parser.add_argument('--noise_schedule', type=str, choices=['linear', 'cosine'], default=None,
                        help='Noise schedule type (overrides config)')
    parser.add_argument('--min_noise_level', type=float, default=None,
                        help='Minimum noise level to start with (overrides config)')
    
    # Variables for variable length and max audio length
    parser.add_argument('--variable_length', action='store_true',
                        help='Enable variable length processing')
    parser.add_argument('--max_audio_length', type=float, default=None,
                        help='Maximum audio length in seconds')
    
    # Argument for mixed precision training
    parser.add_argument('--precision', type=str, choices=['32-true', '16-mixed', 'bf16-mixed'],
                        default=None, help='Precision for training (overrides config)')
    
    # Arguments for gradient accumulation
    parser.add_argument('--accumulate_grad_batches', type=int, default=None,
                        help='Number of batches to accumulate gradients (overrides config)')
    
    # Load from pretrained model
    parser.add_argument('--load_decoder', type=str, default=None,
                        help='Path to pretrained model to load decoder weights from')
    
    args = parser.parse_args()
    
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
    
    if args.data_key:
        config['data']['data_key'] = args.data_key
    
    if args.max_samples:
        config['data']['max_samples'] = args.max_samples
    
    # Apply command line overrides for noise conditioning parameters
    if args.noise_ramp_steps is not None:
        config['conditioning']['noise_ramp_steps'] = args.noise_ramp_steps
    
    if args.noise_schedule:
        config['conditioning']['noise_schedule'] = args.noise_schedule
    
    if args.min_noise_level is not None:
        config['conditioning']['min_noise_level'] = args.min_noise_level
    
    # Set variable length mode
    if args.variable_length:
        config['data']['variable_length'] = True
        config['model']['variable_length_mode'] = True
    
    # Set maximum audio length
    if args.max_audio_length:
        config['audio']['max_audio_length'] = args.max_audio_length
    
    # Set precision
    if args.precision:
        config['train']['precision'] = args.precision
    
    # Set gradient accumulation
    if args.accumulate_grad_batches:
        config['train']['accumulate_grad_batches'] = args.accumulate_grad_batches
    
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
    print(f"  - Accumulated batches: {config['train'].get('accumulate_grad_batches', 1)}")
    print(f"  - Effective batch size: {batch_size * config['train'].get('accumulate_grad_batches', 1)}")
    print(f"  - Variable length mode: {config['data'].get('variable_length', False)}")
    print(f"  - Noise ramp steps: {config['conditioning'].get('noise_ramp_steps', 10000)}")
    print(f"  - Min noise level: {config['conditioning'].get('min_noise_level', 0.0)}")
    print(f"  - Precision: {config['train'].get('precision', '32-true')}")
    
    # Setup TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=save_dir,
        name='lightning_logs'
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(save_dir, 'checkpoints'),
            filename='conditional-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val_loss',
            patience=100,
            mode='min'
        ),
    ]
    
    # Create model
    model = ConditionalMultiBandUNet(config)
    
    # Load decoder weights from pretrained model if specified
    if args.load_decoder:
        print(f"Loading decoder weights from {args.load_decoder}")
        pretrained_state = torch.load(args.load_decoder, map_location='cpu')
        
        if 'state_dict' in pretrained_state:
            pretrained_state = pretrained_state['state_dict']
        
        # Filter to get only decoder weights
        decoder_state = {}
        for key, value in pretrained_state.items():
            if key.startswith('unet.decoder') or key.startswith('decoder'):
                # Remove 'unet.' prefix if present for compatibility
                if key.startswith('unet.'):
                    new_key = 'unet.' + key[5:]
                else:
                    new_key = 'unet.' + key
                decoder_state[new_key] = value
        
        # Load the decoder weights
        model.load_state_dict(decoder_state, strict=False)
        print(f"Loaded {len(decoder_state)} decoder parameters from pretrained model")
    
    # Create data module
    data_module = ConditionalDataModule(config)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config['train']['num_epochs'],
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config['train'].get('log_interval', 10),
        deterministic=False,  # Set to False for better performance
        accelerator='auto',
        devices='auto',
        precision=config['train'].get('precision', '32-true'),
        accumulate_grad_batches=config['train'].get('accumulate_grad_batches', 1),
        gradient_clip_val=1.0,  # Add gradient clipping for stability
        benchmark=True  # Enable cudnn benchmarking for faster training
    )
    
    try:
        trainer.fit(model, data_module, ckpt_path=args.resume)
        print(f"Training completed. Best model saved with val_loss: {trainer.callback_metrics.get('val_loss', 0):.6f}")
    finally:
        # Clean up H5 files
        H5FileManager.get_instance().close_all()
    
if __name__ == "__main__":
    main()