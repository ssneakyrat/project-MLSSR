import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa

from models.multi_band_unet import MultiBandUNet
from models.hifi_gan import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from models.hifi_losses import HiFiGANLoss


class MultiBandUNetWithHiFiGAN(pl.LightningModule):
    """Combined model with MultiBandUNet for mel-spectrogram processing and HiFi-GAN for vocoding"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Set manual optimization for multiple optimizers
        self.automatic_optimization = False
        
        # Initialize U-Net model
        self.unet = MultiBandUNet(config)
        
        # Initialize HiFi-GAN components if vocoder is enabled
        self.vocoder_enabled = config.get('vocoder', {}).get('enabled', False)
        
        # Set up manual gradient accumulation
        self.accumulate_grad_batches = config['train'].get('accumulate_grad_batches', 1)
        self.current_accumulation_step = 0
        
        if self.vocoder_enabled:
            self._init_vocoder(config)
            
            # Training settings
            self.joint_training = config.get('vocoder', {}).get('joint_training', True)
            self.disc_start_step = config.get('vocoder', {}).get('disc_start_step', 0)
            self.discriminator_train_every = config.get('vocoder', {}).get('disc_train_every', 1)
            self.generator_train_every = config.get('vocoder', {}).get('gen_train_every', 1)
            
            # Set up mel spectrogram extraction parameters for loss calculation
            self.mel_config = {
                'sample_rate': config['audio']['sample_rate'],
                'n_fft': config['audio']['n_fft'],
                'hop_length': config['audio']['hop_length'],
                'win_length': config['audio']['win_length'],
                'n_mels': config['audio']['n_mels'],
                'fmin': config['audio']['fmin'],
                'fmax': config['audio']['fmax']
            }
            
            # Loss weights from config
            lambda_adv = config.get('vocoder', {}).get('lambda_adv', 1.0)
            lambda_fm = config.get('vocoder', {}).get('lambda_fm', 2.0)
            lambda_mel = config.get('vocoder', {}).get('lambda_mel', 45.0)
            
            # Initialize combined loss function
            self.hifi_loss = HiFiGANLoss(
                lambda_adv=lambda_adv,
                lambda_fm=lambda_fm,
                lambda_mel=lambda_mel,
                mel_config=self.mel_config
            )
            
            # Total number of model parameters
            unet_params = sum(p.numel() for p in self.unet.parameters())
            gen_params = sum(p.numel() for p in self.generator.parameters())
            disc_params = sum(p.numel() for p in list(self.mpd.parameters()) + list(self.msd.parameters()))
            
            print(f"MultiBandUNet parameters: {unet_params:,}")
            print(f"HiFi-GAN Generator parameters: {gen_params:,}")
            print(f"HiFi-GAN Discriminator parameters: {disc_params:,}")
            print(f"Total parameters: {unet_params + gen_params + disc_params:,}")
            
            if self.accumulate_grad_batches > 1:
                print(f"Using manual gradient accumulation with {self.accumulate_grad_batches} steps")
        else:
            print("Running in mel-spectrogram-only mode (vocoder disabled)")
            # Count only U-Net parameters
            unet_params = sum(p.numel() for p in self.unet.parameters())
            print(f"MultiBandUNet parameters: {unet_params:,}")
        
        # Register metrics
        self.train_step_idx = 0
    
    def _init_vocoder(self, config):
        """Initialize the HiFi-GAN vocoder components"""
        vocoder_config = config.get('vocoder', {})
        
        # Generator parameters
        upsample_initial_channel = vocoder_config.get('upsample_initial_channel', 128)
        upsample_rates = vocoder_config.get('upsample_rates', [8, 8, 2, 2])
        upsample_kernel_sizes = vocoder_config.get('upsample_kernel_sizes', [16, 16, 4, 4])
        resblock_kernel_sizes = vocoder_config.get('resblock_kernel_sizes', [3, 7, 11])
        resblock_dilation_sizes = vocoder_config.get('resblock_dilation_sizes', 
                                                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
        
        # Initialize generator
        self.generator = Generator(
            in_channels=config['model']['mel_bins'],
            upsample_initial_channel=upsample_initial_channel,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes
        )
        
        # Discriminator parameters
        disc_periods = vocoder_config.get('disc_periods', [2, 3, 5, 7, 11])
        disc_channels = vocoder_config.get('disc_channels', 16)
        disc_max_channels = vocoder_config.get('disc_max_channels', 256)
        
        # Initialize discriminators
        self.mpd = MultiPeriodDiscriminator(
            periods=disc_periods,
            channels=disc_channels,
            max_channels=disc_max_channels
        )
        
        self.msd = MultiScaleDiscriminator(
            channels=disc_channels,
            max_channels=disc_max_channels
        )
        
        # Separate checkpointing mode
        self.separate_checkpointing = vocoder_config.get('separate_checkpointing', False)
        
        # Set learning rates
        self.gen_lr = vocoder_config.get('gen_lr', config['train'].get('learning_rate', 0.0002))
        self.disc_lr = vocoder_config.get('disc_lr', config['train'].get('learning_rate', 0.0002))
        
        print(f"HiFi-GAN vocoder initialized with:")
        print(f"  - Upsampling rates: {upsample_rates} (total upsampling: {np.prod(upsample_rates)})")
        print(f"  - Initial channels: {upsample_initial_channel}")
        print(f"  - Learning rates: G={self.gen_lr}, D={self.disc_lr}")
    
    def forward(self, mel_input):
        """
        Forward pass through both U-Net and HiFi-GAN (if enabled)
        
        Args:
            mel_input: Input mel spectrogram [B, 1, F, T]
            
        Returns:
            dict containing:
                - 'mel_output': Reconstructed mel spectrogram from U-Net [B, 1, F, T]
                - 'audio_output': Generated waveform (if vocoder enabled) [B, 1, T']
        """
        # Process with U-Net
        mel_output = self.unet(mel_input)
        
        result = {'mel_output': mel_output}
        
        # Generate waveform with HiFi-GAN if enabled
        if self.vocoder_enabled:
            # Prepare mel for generator: [B, 1, F, T] -> [B, F, T]
            mel_for_generator = mel_output.squeeze(1)
            
            # Generate waveform
            audio_output = self.generator(mel_for_generator)
            result['audio_output'] = audio_output
        
        return result
    
    def training_step(self, batch, batch_idx):
        """
        Training step with manual optimization and gradient accumulation
        
        Args:
            batch: Input batch which is either:
                - mel_spec [B, 1, F, T] (if vocoder disabled)
                - (mel_spec [B, 1, F, T], audio [B, 1, T']) (if vocoder enabled)
            batch_idx: Batch index
        """
        self.train_step_idx += 1
        
        # Update accumulation step counter
        self.current_accumulation_step = (self.current_accumulation_step + 1) % self.accumulate_grad_batches
        is_last_accumulation_step = (self.current_accumulation_step == 0) or (self.current_accumulation_step == self.accumulate_grad_batches)
        
        # Calculate normalization factor for loss when accumulating gradients
        accumulation_factor = 1.0 / self.accumulate_grad_batches if self.accumulate_grad_batches > 1 else 1.0
        
        if not self.vocoder_enabled:
            # Standard U-Net training with single optimizer
            opt = self.optimizers()
            
            # Process input data - handle list structure
            mel_input = batch
            
            # Handle batch being a list or tuple
            if isinstance(mel_input, (list, tuple)):
                if len(mel_input) > 0:
                    if isinstance(mel_input[0], torch.Tensor):
                        mel_input = mel_input[0]  # Use first element if it's a tensor
                    elif isinstance(mel_input, tuple) and len(mel_input) == 2:
                        # This might be (data, mask) format
                        mel_input = mel_input[0]
            
            # Forward pass
            y_pred = self(mel_input)
            mel_output = y_pred['mel_output']
            
            # Loss calculation with normalization factor
            loss = self.unet.loss_fn(mel_output, mel_input) * accumulation_factor
            
            # Manual gradient accumulation and optimization
            self.manual_backward(loss)
            
            # Only update weights on the last accumulation step
            if is_last_accumulation_step:
                opt.step()
                opt.zero_grad()
            
            # Log metrics
            self.log('train_loss', loss / accumulation_factor, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
            return loss
        else:
            # Joint training with multiple optimizers (generator and discriminator)
            # Get optimizers
            opt_g, opt_d = self.optimizers()
            
            # Handle input data format
            if isinstance(batch, (tuple, list)):
                # Check if the first element is itself a tuple/list (complex structure)
                if len(batch) == 2 and isinstance(batch[0], (tuple, list)):
                    # Complex structure with masks: ((mel, audio), (mel_mask, audio_mask))
                    data, masks = batch
                    mel_input, audio_target = data
                else:
                    # Simple structure: (mel, audio)
                    mel_input, audio_target = batch
            else:
                # This should not happen in vocoder mode
                print(f"Warning: Expected tuple/list batch in vocoder mode, got {type(batch)}")
                return None
            
            # Ensure inputs are tensors, not lists
            if isinstance(mel_input, list):
                print(f"Warning: mel_input is a list with {len(mel_input)} elements in training_step")
                if len(mel_input) > 0 and isinstance(mel_input[0], torch.Tensor):
                    mel_input = mel_input[0]
                else:
                    print(f"Cannot process list input for mel_input")
                    # Return a dummy value to avoid crashing
                    return torch.tensor(0.0, device=self.device)
            
            if isinstance(audio_target, list):
                print(f"Warning: audio_target is a list with {len(audio_target)} elements in training_step")
                if len(audio_target) > 0 and isinstance(audio_target[0], torch.Tensor):
                    audio_target = audio_target[0]
                else:
                    print(f"Cannot process list input for audio_target")
                    # Return a dummy value to avoid crashing
                    return torch.tensor(0.0, device=self.device)
            
            # ========== Step 1: Train Generator (U-Net + HiFi-GAN Generator) ==========
            # Only train generator on specified intervals
            if (self.train_step_idx % self.generator_train_every) == 0:
                # Forward pass for generator
                outputs = self(mel_input)
                mel_output = outputs['mel_output']
                audio_output = outputs['audio_output']
                
                # U-Net loss
                unet_loss = self.unet.loss_fn(mel_output, mel_input)
                
                # Run discriminators on generated audio
                mpd_fake_feat_maps, mpd_fake_scores = self.mpd(audio_output)
                msd_fake_feat_maps, msd_fake_scores = self.msd(audio_output)
                
                # Get feature maps from real audio
                with torch.no_grad():
                    mpd_real_feat_maps, _ = self.mpd(audio_target)
                    msd_real_feat_maps, _ = self.msd(audio_target)
                
                # Combine feature maps and scores
                real_feat_maps = mpd_real_feat_maps + msd_real_feat_maps
                fake_feat_maps = mpd_fake_feat_maps + msd_fake_feat_maps
                fake_scores = mpd_fake_scores + msd_fake_scores
                
                # Calculate generator losses
                gen_loss, gen_losses_dict = self.hifi_loss.generator_loss(
                    fake_scores, real_feat_maps, fake_feat_maps, audio_target, audio_output
                )
                
                # Combine losses with weights from config
                unet_weight = self.config.get('vocoder', {}).get('unet_weight', 1.0)
                vocoder_weight = self.config.get('vocoder', {}).get('vocoder_weight', 1.0)
                total_g_loss = (unet_weight * unet_loss + vocoder_weight * gen_loss) * accumulation_factor
                
                # Manual backward pass for gradient accumulation
                self.manual_backward(total_g_loss)
                
                # Only update weights on the last accumulation step
                if is_last_accumulation_step:
                    opt_g.step()
                    opt_g.zero_grad()
                
                # Log metrics (unscaled)
                self.log('train_unet_loss', unet_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log('train_gen_loss', gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                self.log('train_total_g_loss', total_g_loss / accumulation_factor, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                
                for k, v in gen_losses_dict.items():
                    self.log(f'train_{k}', v, on_step=False, on_epoch=True, logger=True)
            
            # ========== Step 2: Train Discriminator ==========
            # Only train discriminator after disc_start_step and on specified intervals
            if self.train_step_idx >= self.disc_start_step and (self.train_step_idx % self.discriminator_train_every) == 0:
                # Generate audio without gradients for discriminator training
                with torch.no_grad():
                    outputs = self(mel_input)
                    audio_output = outputs['audio_output']
                
                # MPD discriminator
                mpd_real_feat_maps, mpd_real_scores = self.mpd(audio_target)
                mpd_fake_feat_maps, mpd_fake_scores = self.mpd(audio_output.detach())
                
                # MSD discriminator
                msd_real_feat_maps, msd_real_scores = self.msd(audio_target)
                msd_fake_feat_maps, msd_fake_scores = self.msd(audio_output.detach())
                
                # Combine scores
                real_scores = mpd_real_scores + msd_real_scores
                fake_scores = mpd_fake_scores + msd_fake_scores
                
                # Calculate discriminator loss with accumulation factor
                disc_loss, disc_losses_dict = self.hifi_loss.discriminator_loss(
                    real_scores, fake_scores
                )
                disc_loss = disc_loss * accumulation_factor
                
                # Manual backward pass for gradient accumulation
                self.manual_backward(disc_loss)
                
                # Only update weights on the last accumulation step
                if is_last_accumulation_step:
                    opt_d.step()
                    opt_d.zero_grad()
                
                # Log metrics (unscaled)
                self.log('train_disc_loss', disc_loss / accumulation_factor, on_step=True, on_epoch=True, prog_bar=True, logger=True)
                for k, v in disc_losses_dict.items():
                    self.log(f'train_{k}', v, on_step=False, on_epoch=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        """Validation step for combined model"""
        if not self.vocoder_enabled:
            # U-Net only validation
            mel_input = batch
            
            # Handle batch being a list or tuple
            if isinstance(mel_input, (list, tuple)):
                if len(mel_input) > 0:
                    if isinstance(mel_input[0], torch.Tensor):
                        mel_input = mel_input[0]  # Use first element if it's a tensor
                    elif isinstance(mel_input, tuple) and len(mel_input) == 2:
                        # This might be (data, mask) format
                        mel_input = mel_input[0]
            
            # Forward pass through U-Net
            outputs = self(mel_input)
            mel_output = outputs['mel_output']
            
            # Calculate U-Net loss
            val_loss = self.unet.loss_fn(mel_output, mel_input)
            
            # Log metrics
            self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            # Log validation images
            if batch_idx == 0:
                self._log_validation_spectrograms(mel_input, mel_output)
            
            return val_loss
        else:
            # Joint validation with vocoder
            if isinstance(batch, (tuple, list)):
                # Check if the first element is itself a tuple/list (complex structure)
                if len(batch) == 2 and isinstance(batch[0], (tuple, list)):
                    # Complex structure with masks: ((mel, audio), (mel_mask, audio_mask))
                    data, masks = batch
                    mel_input, audio_target = data
                else:
                    # Simple structure: (mel, audio)
                    mel_input, audio_target = batch
            else:
                # Fallback to U-Net only
                return super().validation_step(batch, batch_idx)
            
            # Ensure inputs are tensors, not lists
            if isinstance(mel_input, list):
                print(f"Warning: mel_input is a list with {len(mel_input)} elements in validation_step")
                if len(mel_input) > 0 and isinstance(mel_input[0], torch.Tensor):
                    mel_input = mel_input[0]
                else:
                    print(f"Cannot process list input for mel_input")
                    # Return a dummy value to avoid crashing
                    return torch.tensor(0.0, device=self.device)
            
            if isinstance(audio_target, list):
                print(f"Warning: audio_target is a list with {len(audio_target)} elements in validation_step")
                if len(audio_target) > 0 and isinstance(audio_target[0], torch.Tensor):
                    audio_target = audio_target[0]
                else:
                    print(f"Cannot process list input for audio_target")
                    # Return a dummy value to avoid crashing
                    return torch.tensor(0.0, device=self.device)
            
            # Forward pass
            outputs = self(mel_input)
            mel_output = outputs['mel_output']
            audio_output = outputs['audio_output']
            
            # U-Net loss
            unet_loss = self.unet.loss_fn(mel_output, mel_input)
            
            # Run discriminators
            mpd_real_feat_maps, mpd_real_scores = self.mpd(audio_target)
            mpd_fake_feat_maps, mpd_fake_scores = self.mpd(audio_output)
            
            msd_real_feat_maps, msd_real_scores = self.msd(audio_target)
            msd_fake_feat_maps, msd_fake_scores = self.msd(audio_output)
            
            # Combine feature maps and scores
            real_feat_maps = mpd_real_feat_maps + msd_real_feat_maps
            fake_feat_maps = mpd_fake_feat_maps + msd_fake_feat_maps
            
            real_scores = mpd_real_scores + msd_real_scores
            fake_scores = mpd_fake_scores + msd_fake_scores
            
            # Generator loss
            gen_loss, gen_losses_dict = self.hifi_loss.generator_loss(
                fake_scores, real_feat_maps, fake_feat_maps, audio_target, audio_output
            )
            
            # Discriminator loss
            disc_loss, disc_losses_dict = self.hifi_loss.discriminator_loss(
                real_scores, fake_scores
            )
            
            # Combine losses
            unet_weight = self.config.get('vocoder', {}).get('unet_weight', 1.0)
            vocoder_weight = self.config.get('vocoder', {}).get('vocoder_weight', 1.0)
            
            total_loss = unet_weight * unet_loss + vocoder_weight * gen_loss
            
            # Log metrics
            self.log('val_unet_loss', unet_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_gen_loss', gen_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_disc_loss', disc_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            # Log validation images and audio
            if batch_idx == 0:
                self._log_validation_spectrograms(mel_input, mel_output)
                self._log_validation_audio(audio_target, audio_output)
            
            return total_loss
    
    def _log_validation_spectrograms(self, mel_input, mel_output):
        """Log mel-spectrograms for validation"""
        # Get validation config
        val_config = self.config.get('validation', {})
        max_samples = min(val_config.get('max_samples', 4), mel_input.size(0))
        
        # Select random samples
        indices = torch.randperm(mel_input.size(0))[:max_samples]
        
        for i, idx in enumerate(indices):
            input_mel = mel_input[idx, 0].cpu().numpy()
            output_mel = mel_output[idx, 0].cpu().numpy()
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            im1 = ax1.imshow(input_mel, origin='lower', aspect='auto', cmap='viridis')
            ax1.set_title('Input Mel Spectrogram')
            ax1.set_xlabel('Time Frames')
            ax1.set_ylabel('Mel Bins')
            fig.colorbar(im1, ax=ax1)
            
            im2 = ax2.imshow(output_mel, origin='lower', aspect='auto', cmap='viridis')
            ax2.set_title('Reconstructed Mel Spectrogram')
            ax2.set_xlabel('Time Frames')
            ax2.set_ylabel('Mel Bins')
            fig.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            
            # Convert to image and log
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = Image.open(buf)
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)
            
            self.logger.experiment.add_image(f'val_sample_{i}', img_tensor, self.current_epoch)
            
            plt.close(fig)
    
    def _log_validation_audio(self, audio_target, audio_output, max_samples=2):
        """Log audio samples for validation"""
        if not self.vocoder_enabled:
            return
        
        # Create a single fixed directory for audio samples (no epoch-specific subdirectories)
        audio_dir = os.path.join(self.logger.log_dir if hasattr(self.logger, 'log_dir') else 'validation_audio', 
                                'latest_samples')
        os.makedirs(audio_dir, exist_ok=True)
        
        # Select random samples
        max_samples = min(max_samples, audio_target.size(0))
        indices = torch.randperm(audio_target.size(0))[:max_samples]
        
        sample_rate = self.config['audio']['sample_rate']
        
        for i, idx in enumerate(indices):
            # Get target and output audio
            target = audio_target[idx, 0].cpu().numpy()
            output = audio_output[idx, 0].cpu().numpy()
            
            # Ensure audio is float32 (soundfile doesn't support float16)
            target = target.astype(np.float32)
            output = output.astype(np.float32)
            
            # Normalize to [-1, 1]
            target = target / (np.abs(target).max() + 1e-7)
            output = output / (np.abs(output).max() + 1e-7)
            
            # Save to files with proper paths - using fixed names to ensure only latest files are kept
            target_path = os.path.join(audio_dir, f'val_target_{i}.wav')
            output_path = os.path.join(audio_dir, f'val_output_{i}.wav')
            
            # Specify dtype explicitly when writing audio
            sf.write(target_path, target, sample_rate, subtype='FLOAT')
            sf.write(output_path, output, sample_rate, subtype='FLOAT')
            
            # Log to tensorboard
            self.logger.experiment.add_audio(
                f'val_target_{i}', target.reshape(1, -1), self.current_epoch, sample_rate
            )
            self.logger.experiment.add_audio(
                f'val_output_{i}', output.reshape(1, -1), self.current_epoch, sample_rate
            )
    
    def configure_optimizers(self):
        """Configure optimizers for all components"""
        if not self.vocoder_enabled:
            # Standard optimizer from U-Net
            return self.unet.configure_optimizers()
        else:
            # Joint training with multiple optimizers
            learning_rate = self.config['train'].get('learning_rate', 0.001)
            weight_decay = self.config['train'].get('weight_decay', 0.0001)
            
            # Get vocoder learning rates (if specified)
            gen_lr = self.gen_lr
            disc_lr = self.disc_lr
            
            # Joint optimizer for U-Net and Generator
            optimizer_g = torch.optim.AdamW(
                list(self.unet.parameters()) + list(self.generator.parameters()),
                lr=gen_lr,
                weight_decay=weight_decay
            )
            
            # Separate optimizer for discriminators
            optimizer_d = torch.optim.AdamW(
                list(self.mpd.parameters()) + list(self.msd.parameters()),
                lr=disc_lr,
                weight_decay=weight_decay
            )
            
            # Configure schedulers - now just return the optimizers without schedulers
            # For manual optimization, schedulers would need to be updated manually
            return [optimizer_g, optimizer_d]
    
    def _get_scheduler(self, optimizer):
        """Get scheduler for the specified optimizer"""
        scheduler_type = self.config['train'].get('lr_scheduler', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            lr_patience = self.config['train'].get('lr_patience', 5)
            lr_factor = self.config['train'].get('lr_factor', 0.5)
            
            return {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=lr_factor,
                    patience=lr_patience,
                    verbose=True
                ),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )
        else:
            return None
            
    def on_save_checkpoint(self, checkpoint):
        """Custom checkpoint handling"""
        if self.vocoder_enabled and self.separate_checkpointing:
            # Get a directory for saving component checkpoints
            save_dir = os.path.join(self.logger.log_dir if hasattr(self.logger, 'log_dir') else 'checkpoints', 
                                'component_checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            
            # Create a combined checkpoint for all components: UNet, generator, and discriminators
            combined_state = {
                'unet': self.unet.state_dict(),
                'generator': self.generator.state_dict(),
                'mpd': self.mpd.state_dict(),
                'msd': self.msd.state_dict(),
                'epoch': self.current_epoch,
                'val_loss': self.trainer.callback_metrics.get('val_loss', float('inf')) if hasattr(self, 'trainer') else float('inf')
            }
            
            # Always save a copy with the current best
            current_val_loss = combined_state['val_loss']
            if not hasattr(self, '_best_val_loss') or current_val_loss < getattr(self, '_best_val_loss', float('inf')):
                self._best_val_loss = current_val_loss
                best_filepath = os.path.join(save_dir, "complete_model_best.pt")
                torch.save(combined_state, best_filepath)
                #print(f"New best model (val_loss={current_val_loss:.6f})! Saved all components to {best_filepath}")
            
            #print(f"Saved complete model checkpoint (UNet + vocoder) to TensorBoard directory")
    
    def on_load_checkpoint(self, checkpoint):
        """Custom checkpoint loading"""
        if self.vocoder_enabled and self.separate_checkpointing:
            # Check if separate component checkpoints exist and load them if available
            unet_path = f"unet_checkpoint_epoch_{checkpoint['epoch']}.pt"
            gen_path = f"generator_checkpoint_epoch_{checkpoint['epoch']}.pt"
            mpd_path = f"mpd_checkpoint_epoch_{checkpoint['epoch']}.pt"
            msd_path = f"msd_checkpoint_epoch_{checkpoint['epoch']}.pt"
            
            if all(map(os.path.exists, [unet_path, gen_path, mpd_path, msd_path])):
                # Load separate checkpoints
                self.unet.load_state_dict(torch.load(unet_path))
                self.generator.load_state_dict(torch.load(gen_path))
                self.mpd.load_state_dict(torch.load(mpd_path))
                self.msd.load_state_dict(torch.load(msd_path))
                
                print(f"Loaded separate component checkpoints from epoch {checkpoint['epoch']}")
                
                # Skip standard loading
                return True
        
        # Default behavior
        return None