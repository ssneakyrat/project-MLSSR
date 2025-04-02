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
import wandb

from models.multi_band_unet import MultiBandUNet
from models.hifi_gan import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from models.hifi_losses import HiFiGANLoss


class MultiBandUNetWithHiFiGAN(pl.LightningModule):
    """Combined model with MultiBandUNet for mel-spectrogram processing and HiFi-GAN for vocoding"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize U-Net model
        self.unet = MultiBandUNet(config)
        
        # Initialize HiFi-GAN components if vocoder is enabled
        self.vocoder_enabled = config.get('vocoder', {}).get('enabled', False)
        
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
    
    def training_step(self, batch, batch_idx, optimizer_idx=None):
        """
        Training step supporting both U-Net and HiFi-GAN training
        
        Args:
            batch: Input batch which is either:
                  - mel_spec [B, 1, F, T] (if vocoder disabled)
                  - (mel_spec [B, 1, F, T], audio [B, 1, T']) (if vocoder enabled)
            batch_idx: Batch index
            optimizer_idx: Which optimizer to use
                          - None if vocoder disabled
                          - 0 for U-Net and Generator
                          - 1 for Discriminators
        """
        self.train_step_idx += 1
        
        if not self.vocoder_enabled:
            # Standard U-Net training
            return self._training_step_unet_only(batch, batch_idx)
        else:
            # Joint training with HiFi-GAN
            if isinstance(batch, tuple) or isinstance(batch, list):
                mel_input, audio_target = batch
            else:
                # This should not happen in vocoder mode
                mel_input = batch
                audio_target = None
                print(f"Warning: Expected tuple/list batch in vocoder mode, got {type(batch)}")
                # Fallback to U-Net only
                return self._training_step_unet_only(mel_input, batch_idx)
            
            # Check if we need to skip discriminator training based on step
            if optimizer_idx == 1 and self.train_step_idx < self.disc_start_step:
                # Return dummy loss
                return torch.tensor(0.0, requires_grad=True, device=self.device)
            
            # Check training schedule
            if optimizer_idx == 0:  # U-Net + Generator
                if (self.train_step_idx % self.generator_train_every) != 0:
                    return torch.tensor(0.0, requires_grad=True, device=self.device)
                return self._training_step_generator(mel_input, audio_target, batch_idx)
            elif optimizer_idx == 1:  # Discriminator
                if (self.train_step_idx % self.discriminator_train_every) != 0:
                    return torch.tensor(0.0, requires_grad=True, device=self.device)
                return self._training_step_discriminator(mel_input, audio_target, batch_idx)
            else:
                raise ValueError(f"Invalid optimizer_idx: {optimizer_idx}")
    
    def _training_step_unet_only(self, mel_input, batch_idx):
        """U-Net only training step (no vocoder)"""
        # Forward pass through U-Net
        outputs = self(mel_input)
        mel_output = outputs['mel_output']
        
        # Calculate U-Net loss
        loss = self.unet.loss_fn(mel_output, mel_input)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def _training_step_generator(self, mel_input, audio_target, batch_idx):
        """Generator training step (U-Net + HiFi-GAN Generator)"""
        # Forward pass through both models
        outputs = self(mel_input)
        mel_output = outputs['mel_output']
        audio_output = outputs['audio_output']
        
        # U-Net mel reconstruction loss
        unet_loss = self.unet.loss_fn(mel_output, mel_input)
        
        # Run discriminators on real and generated audio
        with torch.no_grad():
            # MPD
            mpd_real_feat_maps, mpd_real_scores = self.mpd(audio_target)
            
            # MSD
            msd_real_feat_maps, msd_real_scores = self.msd(audio_target)
        
        # Discriminate generated audio
        mpd_fake_feat_maps, mpd_fake_scores = self.mpd(audio_output)
        msd_fake_feat_maps, msd_fake_scores = self.msd(audio_output)
        
        # Combine feature maps and scores from both discriminators
        real_feat_maps = mpd_real_feat_maps + msd_real_feat_maps
        fake_feat_maps = mpd_fake_feat_maps + msd_fake_feat_maps
        
        fake_scores = mpd_fake_scores + msd_fake_scores
        
        # Calculate generator losses
        gen_loss, gen_losses_dict = self.hifi_loss.generator_loss(
            fake_scores, real_feat_maps, fake_feat_maps, audio_target, audio_output
        )
        
        # Combine U-Net and generator losses
        # Use a weighting factor (can be adjusted in config)
        unet_weight = self.config.get('vocoder', {}).get('unet_weight', 1.0)
        vocoder_weight = self.config.get('vocoder', {}).get('vocoder_weight', 1.0)
        
        total_loss = unet_weight * unet_loss + vocoder_weight * gen_loss
        
        # Log metrics
        self.log('train_unet_loss', unet_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_gen_loss', gen_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        for k, v in gen_losses_dict.items():
            self.log(f'train_{k}', v, on_step=False, on_epoch=True, logger=True)
        
        return total_loss
    
    def _training_step_discriminator(self, mel_input, audio_target, batch_idx):
        """Discriminator training step"""
        # Generate audio without gradients for discriminator training
        with torch.no_grad():
            outputs = self(mel_input)
            audio_output = outputs['audio_output']
        
        # MPD
        mpd_real_feat_maps, mpd_real_scores = self.mpd(audio_target)
        mpd_fake_feat_maps, mpd_fake_scores = self.mpd(audio_output.detach())
        
        # MSD
        msd_real_feat_maps, msd_real_scores = self.msd(audio_target)
        msd_fake_feat_maps, msd_fake_scores = self.msd(audio_output.detach())
        
        # Combine scores from both discriminators
        real_scores = mpd_real_scores + msd_real_scores
        fake_scores = mpd_fake_scores + msd_fake_scores
        
        # Calculate discriminator loss
        disc_loss, disc_losses_dict = self.hifi_loss.discriminator_loss(
            real_scores, fake_scores
        )
        
        # Log metrics
        self.log('train_disc_loss', disc_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for k, v in disc_losses_dict.items():
            self.log(f'train_{k}', v, on_step=False, on_epoch=True, logger=True)
        
        return disc_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for combined model"""
        if not self.vocoder_enabled:
            # U-Net only validation
            mel_input = batch
            
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
            if isinstance(batch, tuple) or isinstance(batch, list):
                mel_input, audio_target = batch
            else:
                # Fallback to U-Net only
                return super().validation_step(batch, batch_idx)
            
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
            self.log('val_gen_loss', gen_loss, on_step=False, on_epoch=True, logger=True)
            self.log('val_disc_loss', disc_loss, on_step=False, on_epoch=True, logger=True)
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
        
        # Select random samples
        max_samples = min(max_samples, audio_target.size(0))
        indices = torch.randperm(audio_target.size(0))[:max_samples]
        
        sample_rate = self.config['audio']['sample_rate']
        
        for i, idx in enumerate(indices):
            # Get target and output audio
            target = audio_target[idx, 0].cpu().numpy()
            output = audio_output[idx, 0].cpu().numpy()
            
            # Normalize to [-1, 1]
            target = target / (np.abs(target).max() + 1e-7)
            output = output / (np.abs(output).max() + 1e-7)
            
            # Save to temp files and log
            target_path = f'val_target_{i}_{self.current_epoch}.wav'
            output_path = f'val_output_{i}_{self.current_epoch}.wav'
            
            sf.write(target_path, target, sample_rate)
            sf.write(output_path, output, sample_rate)
            
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
            
            # Create optimizer groups based on separate_checkpointing
            if self.separate_checkpointing:
                # Separate optimizers for each component
                unet_optimizer = torch.optim.AdamW(
                    self.unet.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
                
                gen_optimizer = torch.optim.AdamW(
                    self.generator.parameters(),
                    lr=gen_lr,
                    weight_decay=weight_decay
                )
                
                disc_optimizer = torch.optim.AdamW(
                    list(self.mpd.parameters()) + list(self.msd.parameters()),
                    lr=disc_lr,
                    weight_decay=weight_decay
                )
                
                # Combine U-Net and Generator for first optimizer group
                unet_gen_optimizer = {
                    'optimizer': torch.optim.AdamW(
                        list(self.unet.parameters()) + list(self.generator.parameters()),
                        lr=learning_rate,
                        weight_decay=weight_decay
                    ),
                    'lr_scheduler': self._get_scheduler(0)
                }
                
                disc_optimizer_config = {
                    'optimizer': disc_optimizer,
                    'lr_scheduler': self._get_scheduler(1)
                }
                
                return [unet_gen_optimizer, disc_optimizer_config]
            else:
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
                
                # Configure schedulers
                scheduler_g = self._get_scheduler(0)
                scheduler_d = self._get_scheduler(1)
                
                return [
                    {'optimizer': optimizer_g, 'lr_scheduler': scheduler_g},
                    {'optimizer': optimizer_d, 'lr_scheduler': scheduler_d}
                ]
    
    def _get_scheduler(self, optimizer_idx):
        """Get scheduler for the specified optimizer"""
        scheduler_type = self.config['train'].get('lr_scheduler', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            lr_patience = self.config['train'].get('lr_patience', 5)
            lr_factor = self.config['train'].get('lr_factor', 0.5)
            
            return {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizers()[optimizer_idx],
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
                self.optimizers()[optimizer_idx],
                step_size=10,
                gamma=0.5
            )
        else:
            return None
    
    def on_save_checkpoint(self, checkpoint):
        """Custom checkpoint handling"""
        if self.vocoder_enabled and self.separate_checkpointing:
            # Save separate checkpoints for each component
            unet_state = {k: v for k, v in self.unet.state_dict().items()}
            gen_state = {k: v for k, v in self.generator.state_dict().items()}
            mpd_state = {k: v for k, v in self.mpd.state_dict().items()}
            msd_state = {k: v for k, v in self.msd.state_dict().items()}
            
            # Save to separate files
            torch.save(unet_state, f"unet_checkpoint_epoch_{self.current_epoch}.pt")
            torch.save(gen_state, f"generator_checkpoint_epoch_{self.current_epoch}.pt")
            torch.save(mpd_state, f"mpd_checkpoint_epoch_{self.current_epoch}.pt")
            torch.save(msd_state, f"msd_checkpoint_epoch_{self.current_epoch}.pt")
            
            print(f"Saved separate component checkpoints at epoch {self.current_epoch}")
    
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