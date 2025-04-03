import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.utils import load_config
from models.multi_band_unet import MultiBandUNet
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
    
    # Add arguments for two-phase training
    parser.add_argument('--pretrain_unet_epochs', type=int, default=config['train']['pretrain_unet_epoch'], 
                        help='Number of epochs to pretrain only the U-Net before joint training')
    parser.add_argument('--load_pretrained_unet', type=str, default=None,
                        help='Path to pretrained U-Net checkpoint to use instead of pretraining')
    parser.add_argument('--skip_joint_training', action='store_true',
                        help='Skip joint training and only do U-Net pretraining')
    
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
    original_vocoder_enabled = config.get('vocoder', {}).get('enabled', True)
    
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
    
    # First phase: Pretrain U-Net if needed
    pretrained_unet_path = args.load_pretrained_unet
    
    if pretrained_unet_path is None and args.pretrain_unet_epochs > 0:
        print(f"\n{'='*80}\nPHASE 1: PRETRAINING U-NET FOR {args.pretrain_unet_epochs} EPOCHS\n{'='*80}")
        
        # Temporarily disable vocoder for pretraining
        if 'vocoder' not in config:
            config['vocoder'] = {}
        original_vocoder_config = config.get('vocoder', {}).copy()
        config['vocoder']['enabled'] = False
        
        # Set up logger for pretraining phase
        pretrain_logger = TensorBoardLogger(
            save_dir=os.path.join(save_dir, 'pretrain_unet'),
            name='lightning_logs'
        )
        
        # Set up callbacks for pretraining phase
        pretrain_callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(save_dir, 'pretrain_unet', 'checkpoints'),
                filename='unet-{epoch:02d}-{val_loss:.4f}',
                save_top_k=1,
                monitor='val_loss',
                mode='min',
                save_last=True
            ),
            LearningRateMonitor(logging_interval='epoch'),
            EarlyStopping(
                monitor='val_loss',
                patience=5,  # Shorter patience for pretraining
                mode='min'
            ),
        ]
        
        # Create U-Net model for pretraining
        unet_model = MultiBandUNet(config)
        data_module = MelAudioDataModule(config)
        
        # Define pretraining configuration
        pretrain_kwargs = {
            'max_epochs': args.pretrain_unet_epochs,
            'logger': pretrain_logger,
            'callbacks': pretrain_callbacks,
            'log_every_n_steps': config['train'].get('log_interval', 10),
            'deterministic': config.get('train', {}).get('deterministic', False),
            'accelerator': 'auto',
            'devices': 'auto',
            'precision': config['train'].get('precision', '32-true'),
            'benchmark': True,
        }
        
        # Add gradient accumulation if specified
        if 'accumulate_grad_batches' in config['train']:
            pretrain_kwargs['accumulate_grad_batches'] = config['train']['accumulate_grad_batches']
        
        # Create trainer for pretraining
        pretrain_trainer = pl.Trainer(**pretrain_kwargs)
        
        try:
            # Train U-Net only
            pretrain_trainer.fit(unet_model, data_module)
            
            # Get the best checkpoint path
            pretrained_unet_path = pretrain_trainer.checkpoint_callback.best_model_path
            print(f"U-Net pretraining completed. Best model: {pretrained_unet_path}")
            print(f"Validation loss: {pretrain_trainer.callback_metrics.get('val_loss', 0):.6f}")
            
        except Exception as e:
            print(f"Error during U-Net pretraining: {e}")
            # Continue with joint training even if pretraining fails
        finally:
            H5FileManager.get_instance().close_all()
        
        # Restore original vocoder configuration for joint training
        config['vocoder'] = original_vocoder_config
    
    # Skip joint training if requested
    if args.skip_joint_training:
        print("Skipping joint training as requested.")
        return
    
    # Second phase: Joint training with vocoder
    # Skip if vocoder is disabled by user
    if not args.disable_vocoder and original_vocoder_enabled:
        print(f"\n{'='*80}\nPHASE 2: JOINT TRAINING WITH HIFI-GAN VOCODER\n{'='*80}")
        
        # Ensure vocoder is enabled for joint training
        if 'vocoder' not in config:
            config['vocoder'] = {}
        config['vocoder']['enabled'] = True
        
        # Set up logger for joint training
        joint_logger = TensorBoardLogger(
            save_dir=os.path.join(save_dir, 'joint_training'),
            name='lightning_logs'
        )
        
        # Set up callbacks for joint training
        joint_callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(save_dir, 'joint_training', 'checkpoints'),
                filename='combined-{epoch:02d}-{val_loss:.4f}',
                save_top_k=1,
                monitor='val_loss',
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
        
        # Print configuration summary
        max_audio_len = config['audio']['max_audio_length']
        sample_rate = config['audio']['sample_rate']
        hop_length = config['audio']['hop_length']
        batch_size = config['train']['batch_size']
        
        frames_per_second = sample_rate / hop_length
        max_frames = int(max_audio_len * frames_per_second)
        
        print(f"Joint training configuration summary:")
        print(f"  - Maximum audio length: {max_audio_len} seconds")
        print(f"  - Maximum mel frames: {max_frames}")
        print(f"  - Maximum audio samples: {int(max_audio_len * sample_rate)}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Variable length mode: {config['data'].get('variable_length', False)}")
        print(f"  - Vocoder settings:")
        vocoder_config = config['vocoder']
        print(f"    - U-Net weight: {vocoder_config.get('unet_weight', 1.0)}")
        print(f"    - Vocoder weight: {vocoder_config.get('vocoder_weight', 1.0)}")
        print(f"    - Generator channels: {vocoder_config.get('upsample_initial_channel', 128)}")
        print(f"    - Upsampling rates: {vocoder_config.get('upsample_rates', [8, 8, 2, 2])}")
        print(f"    - Discriminator start step: {vocoder_config.get('disc_start_step', 0)}")
        
        # Initialize combined model
        # If we have a pretrained U-Net, load it into the combined model
        combined_model = MultiBandUNetWithHiFiGAN(config)
        
        if pretrained_unet_path:
            print(f"Loading pretrained U-Net weights from: {pretrained_unet_path}")
            # Load the pretrained U-Net weights into the combined model's U-Net component
            pretrained_state_dict = torch.load(pretrained_unet_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in pretrained_state_dict:
                unet_state_dict = pretrained_state_dict['state_dict']
            else:
                unet_state_dict = pretrained_state_dict
                
            # Load the U-Net weights
            combined_model.unet.load_state_dict(unet_state_dict)
            print("Successfully loaded pretrained U-Net weights into combined model")
        
        # Initialize data module for joint training
        data_module = MelAudioDataModule(config)
        
        # Define training configuration
        joint_kwargs = {
            'max_epochs': config['train']['num_epochs'],
            'logger': joint_logger,
            'callbacks': joint_callbacks,
            'log_every_n_steps': config['train'].get('log_interval', 10),
            'deterministic': config.get('train', {}).get('deterministic', False),
            'accelerator': 'auto',
            'devices': 'auto',
            'precision': config['train'].get('precision', '32-true'),
            'benchmark': True,
        }
        
        # Create trainer for joint training
        joint_trainer = pl.Trainer(**joint_kwargs)
        
        try:
            # Train combined model
            joint_trainer.fit(combined_model, data_module, ckpt_path=args.resume)
            print(f"Joint training completed. Best model saved with val_loss: {joint_trainer.callback_metrics.get('val_loss', 0):.6f}")
        except Exception as e:
            print(f"Error during joint training: {e}")
        finally:
            H5FileManager.get_instance().close_all()
    else:
        print("Skipping joint training as vocoder is disabled.")
    
if __name__ == "__main__":
    main()