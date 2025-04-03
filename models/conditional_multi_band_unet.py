import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision

from models.multi_band_unet import MultiBandUNet
from models.conditioning import ConditioningEncoder
from models.noise_strategies import NoiseStrategy, NoiseStepStrategy, CyclicalNoiseStrategy


class NoiseScheduler:
    """Handles noise scheduling during training and sampling"""
    def __init__(self, noise_schedule='linear', noise_steps=1000, min_beta=1e-4, max_beta=0.02):
        self.noise_schedule = noise_schedule
        self.noise_steps = noise_steps
        self.min_beta = min_beta
        self.max_beta = max_beta
        
        # Initialize betas according to schedule
        if noise_schedule == 'linear':
            self.betas = torch.linspace(min_beta, max_beta, noise_steps)
        elif noise_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(noise_steps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")
        
        # Calculate alphas and other constants
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x, t):
        """
        Add noise to the input according to noise schedule
        
        Args:
            x: Input tensor 
            t: Timestep tensor
            
        Returns:
            Noisy tensor
        """
        device = x.device
        
        # Move scheduler parameters to the same device as input
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        
        # Extract alpha values for current timesteps
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
        
        # Make sure shapes are compatible for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
        
        while len(sqrt_one_minus_alphas_cumprod_t.shape) < len(x.shape):
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        # Sample normal noise
        noise = torch.randn_like(x)
        
        # Add noise according to schedule
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def sample_random_noise_level(self, batch_size, device=None):
        """Sample random timesteps for a batch"""
        return torch.randint(0, self.noise_steps, (batch_size,), device=device)


class ConditionalMultiBandUNet(pl.LightningModule):
    """
    Conditional Multi-Band U-Net for mel spectrogram generation
    Uses noise conditioning approach to transition from reconstruction to generation
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create base UNet model with 2 input channels
        self.unet = MultiBandUNet(config, in_channels=2)
        
        # Get the number of phonemes from the config
        num_phonemes = len(config['data'].get('phone_map', []))
        if num_phonemes == 0:
            num_phonemes = 100  # Default if not specified
        
        # Create conditioning encoder
        conditioning_dim = config['conditioning'].get('hidden_dim', 256)
        self.conditioning_encoder = ConditioningEncoder(
            num_phonemes=num_phonemes,
            embedding_dim=config['conditioning'].get('embedding_dim', 256),
            hidden_dim=conditioning_dim
        )
        
        # Create conditioning projector - converts conditioning to the right dimension
        self.conditioning_projector = nn.Conv2d(
            conditioning_dim, 1, kernel_size=3, padding=1
        )
        
        # Create time step embedding
        self.time_embed_dim = config['conditioning'].get('time_embed_dim', 128)
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_embed_dim),
            nn.SiLU(),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
        )
        
        # Create noise scheduler
        self.noise_scheduler = NoiseScheduler(
            noise_schedule=config['conditioning'].get('noise_schedule', 'linear'),
            noise_steps=config['conditioning'].get('noise_steps', 1000),
            min_beta=config['conditioning'].get('min_beta', 1e-4),
            max_beta=config['conditioning'].get('max_beta', 0.02)
        )
        
        # Configure noise strategy
        noise_strategy_type = config['conditioning'].get('noise_strategy', 'step')
        if noise_strategy_type == 'cyclical':
            self.noise_strategy = CyclicalNoiseStrategy(config)
        else:  # 'step' or default
            self.noise_strategy = NoiseStepStrategy(config)
        
        # Training parameters
        self.global_step_counter = 0
        
    def forward(self, x, conditioning=None, timesteps=None):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W] (noisy or mel spectrogram)
            conditioning: Dict containing conditioning features
            timesteps: Noise timesteps [B]
            
        Returns:
            Predicted clean signal or noise (depending on training mode)
        """
        # Handle case where x is a list
        if isinstance(x, list):
            if len(x) > 0 and isinstance(x[0], torch.Tensor):
                x = x[0]
            else:
                # Create a dummy tensor
                x = torch.zeros((1, 1, self.config['model']['mel_bins'], 
                            self.config['model']['time_frames']), 
                            device=self.device)
        
        # Process conditioning if provided
        cond_features = None
        if conditioning is not None:
            mel_shape = (x.size(2), x.size(3))  # (freq_bins, time_frames)
            
            phoneme_ids = conditioning.get('phoneme_ids')
            phoneme_durations = conditioning.get('phoneme_durations')
            midi_pitch = conditioning.get('midi_pitch')
            f0 = conditioning.get('f0')
            
            cond_features = self.conditioning_encoder(
                phoneme_ids, phoneme_durations, midi_pitch, f0, mel_shape
            )
            
            # Project to the right dimension
            cond_features = self.conditioning_projector(cond_features)
        
        # Process time embeddings if provided
        time_emb = None
        if timesteps is not None:
            time_emb = self.time_embed(timesteps.unsqueeze(-1).float() / self.noise_scheduler.noise_steps)
        
        # Create noisy/conditioned input
        if cond_features is not None:
            # Concatenate conditioning along channel dimension
            model_input = torch.cat([x, cond_features], dim=1)
        else:
            model_input = x
        
        # Run UNet model
        output = self.unet(model_input)
        
        return output
    
    def training_step(self, batch, batch_idx):
        """
        Training step with noise conditioning
        
        The model is trained to:
        1. Reconstruct clean spectrograms from noisy ones (early training)
        2. Gradually increase noise to transition to generation
        """
        # Unpack the batch - handle both tuples and lists
        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            mel_spectrograms = batch[0]
            conditioning = batch[1] if len(batch) > 1 else None
            mask = batch[2] if len(batch) > 2 else None
        else:
            mel_spectrograms = batch
            conditioning = None
            mask = None
        
        # Handle case where mel_spectrograms is a list
        if isinstance(mel_spectrograms, list):
            if len(mel_spectrograms) > 0 and isinstance(mel_spectrograms[0], torch.Tensor):
                mel_spectrograms = mel_spectrograms[0]
            else:
                # Create a dummy tensor
                mel_spectrograms = torch.zeros((1, 1, self.config['model']['mel_bins'], 
                                            self.config['model']['time_frames']), 
                                            device=self.device)
        
        # Calculate current noise level using the selected strategy
        noise_level = self.noise_strategy.get_noise_level(self.global_step_counter)
        
        # Choose random noise timesteps for each element in the batch
        batch_size = mel_spectrograms.size(0)
        device = mel_spectrograms.device
        timesteps = self.noise_scheduler.sample_random_noise_level(batch_size, device)
        
        # Apply noise according to schedule
        noisy_mels, noise = self.noise_scheduler.add_noise(mel_spectrograms, timesteps)
        
        # Apply additional adjustments based on current training stage
        if noise_level < 1.0:
            # During transition phase, blend between clean and noisy spectrograms
            model_input = noise_level * noisy_mels + (1.0 - noise_level) * mel_spectrograms
        else:
            # Full noise phase
            model_input = noisy_mels
        
        # Forward pass
        predicted = self(model_input, conditioning, timesteps)
        
        # Calculate loss (predict the original or the noise)
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has right shape for broadcasting
            while len(mask.shape) < len(predicted.shape):
                mask = mask.unsqueeze(1)
            # Only calculate loss on real data (not padding)
            loss = F.mse_loss(predicted * mask, mel_spectrograms * mask, reduction='sum')
            # Normalize by number of non-zero elements
            loss = loss / (mask.sum() + 1e-8)
        else:
            loss = F.mse_loss(predicted, mel_spectrograms)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('noise_level', noise_level, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        self.global_step_counter += 1
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step with both reconstruction and generation evaluation
        """

        # Debug the batch contents
        print(f"Validation batch type: {type(batch)}")
        if isinstance(batch, tuple):
            print(f"Batch is a tuple of length {len(batch)}")
            for i, item in enumerate(batch):
                if i == 1:  # Conditioning data
                    if item is None:
                        print("Conditioning is None")
                    else:
                        print(f"Conditioning keys: {item.keys()}")
                        
        # Unpack the batch - handle both tuples and lists
        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            mel_spectrograms = batch[0]
            conditioning = batch[1] if len(batch) > 1 else None
            mask = batch[2] if len(batch) > 2 else None
        else:
            mel_spectrograms = batch
            conditioning = None
            mask = None
        
        # Handle case where mel_spectrograms is a list
        if isinstance(mel_spectrograms, list):
            if len(mel_spectrograms) > 0 and isinstance(mel_spectrograms[0], torch.Tensor):
                mel_spectrograms = mel_spectrograms[0]
            else:
                # Create a dummy tensor
                mel_spectrograms = torch.zeros((1, 1, self.config['model']['mel_bins'], 
                                            self.config['model']['time_frames']), 
                                            device=self.device)
        
        # Evaluate reconstruction ability
        reconstructed = self(mel_spectrograms, conditioning)
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has right shape for broadcasting
            while len(mask.shape) < len(reconstructed.shape):
                mask = mask.unsqueeze(1)
            # Only calculate loss on real data (not padding)
            recon_loss = F.mse_loss(reconstructed * mask, mel_spectrograms * mask, reduction='sum')
            # Normalize by number of non-zero elements
            recon_loss = recon_loss / (mask.sum() + 1e-8)
        else:
            recon_loss = F.mse_loss(reconstructed, mel_spectrograms)
        
        # Evaluate denoising ability with medium noise
        batch_size = mel_spectrograms.size(0)
        device = mel_spectrograms.device
        timesteps = torch.ones(batch_size, device=device).long() * (self.noise_scheduler.noise_steps // 2)
        noisy_mels, _ = self.noise_scheduler.add_noise(mel_spectrograms, timesteps)
        denoised = self(noisy_mels, conditioning, timesteps)
        
        if mask is not None:
            denoise_loss = F.mse_loss(denoised * mask, mel_spectrograms * mask, reduction='sum')
            denoise_loss = denoise_loss / (mask.sum() + 1e-6)  # Increase from 1e-8
        else:
            denoise_loss = F.mse_loss(denoised, mel_spectrograms)
        
        # Log metrics
        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_denoise_loss', denoise_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', (recon_loss + denoise_loss) / 2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Log validation images if it's the first batch
        if batch_idx == 0:
            self._log_validation_results(mel_spectrograms, reconstructed, denoised, conditioning)
        
        return recon_loss
    
    def _visualize_conditioning(self, conditioning, time_frames):
        """
        Visualize conditioning information
        
        Args:
            conditioning: Dictionary of conditioning tensors
            time_frames: Number of time frames in the mel spectrogram
            
        Returns:
            Matplotlib figure with conditioning visualization
        """
        try:
            # Check what conditioning information is available
            has_phonemes = 'phoneme_ids' in conditioning and conditioning['phoneme_ids'] is not None
            has_midi_pitch = 'midi_pitch' in conditioning and conditioning['midi_pitch'] is not None
            has_f0 = 'f0' in conditioning and conditioning['f0'] is not None
            
            # Count how many plots we need
            plot_count = sum([has_phonemes, has_midi_pitch, has_f0])
            if plot_count == 0:
                return None
            
            fig, axs = plt.subplots(plot_count, 1, figsize=(12, 3 * plot_count))
            
            # Handle single subplot case
            if plot_count == 1:
                axs = [axs]
            
            current_ax = 0
            
            # Plot phoneme information if available
            if has_phonemes:
                ax = axs[current_ax]
                
                # Get phoneme data
                phoneme_ids = conditioning['phoneme_ids'][0].cpu().numpy()
                phoneme_starts = conditioning['phoneme_starts'][0].cpu().numpy()
                phoneme_ends = conditioning['phoneme_ends'][0].cpu().numpy()
                
                # Create a visualization of phoneme segments
                phoneme_viz = np.zeros((50, time_frames))
                for i, (start, end, phoneme_id) in enumerate(zip(phoneme_starts, phoneme_ends, phoneme_ids)):
                    if start < time_frames and end > 0:
                        # Ensure we're within valid range
                        start_idx = max(0, int(start))
                        end_idx = min(time_frames, int(end))
                        if start_idx < end_idx:
                            phoneme_viz[:, start_idx:end_idx] = phoneme_id + 1
                
                # Plot the phoneme segmentation
                im = ax.imshow(phoneme_viz, aspect='auto', origin='lower', cmap='tab20c')
                ax.set_yticks([25])
                ax.set_yticklabels(["Phonemes"])
                ax.set_title("Phoneme Conditioning")
                
                # Add phoneme IDs as text
                prev_end = -100  # To avoid overlapping text
                for i, (start, end, phoneme_id) in enumerate(zip(phoneme_starts, phoneme_ends, phoneme_ids)):
                    if start < time_frames and end > 0:
                        start_idx = max(0, int(start))
                        end_idx = min(time_frames, int(end))
                        if end_idx > start_idx:
                            mid = (start_idx + end_idx) // 2
                            if mid - prev_end > 20:  # Only add text if there's enough space
                                ax.text(mid, 25, str(int(phoneme_id)), ha='center', va='center', fontsize=8,
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                                prev_end = mid
                
                current_ax += 1
            
            # Plot MIDI pitch if available
            if has_midi_pitch:
                ax = axs[current_ax]
                
                midi_pitch = conditioning['midi_pitch'][0].cpu().numpy()
                
                # Plot as a line chart
                ax.plot(np.arange(len(midi_pitch)), midi_pitch, 'b-')
                ax.set_title("MIDI Pitch Conditioning")
                ax.set_ylabel("MIDI Note")
                ax.set_ylim(0, 127)  # MIDI note range
                
                current_ax += 1
            
            # Plot F0 if available
            if has_f0:
                ax = axs[current_ax]
                
                f0 = conditioning['f0'][0].cpu().numpy()
                
                # Plot as a line chart
                ax.plot(np.arange(len(f0)), f0, 'r-')
                ax.set_title("F0 Conditioning")
                ax.set_ylabel("F0 (Hz)")
                ax.set_xlabel("Time Frames")
                
                # Set reasonable y-limits for F0
                max_f0 = max(600, np.max(f0[f0 > 0]) * 1.1) if np.any(f0 > 0) else 600
                ax.set_ylim(0, max_f0)
                
                current_ax += 1
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error visualizing conditioning: {e}")
            return None
        
    def _log_validation_results(self, original, reconstructed, denoised, conditioning=None):
        """Log validation spectrograms, reconstructions and generations"""
        try:
            # Log spectrograms
            max_samples = min(4, original.size(0))
            
            for i in range(max_samples):
                fig, axs = plt.subplots(3, 1, figsize=(10, 12))
                
                # Original
                orig_mel = original[i, 0].cpu().numpy()
                im1 = axs[0].imshow(orig_mel, origin='lower', aspect='auto', cmap='viridis')
                axs[0].set_title('Original Mel Spectrogram')
                axs[0].set_ylabel('Mel Bins')
                plt.colorbar(im1, ax=axs[0])
                
                # Reconstructed
                recon_mel = reconstructed[i, 0].cpu().numpy()
                im2 = axs[1].imshow(recon_mel, origin='lower', aspect='auto', cmap='viridis')
                axs[1].set_title('Reconstructed Mel Spectrogram')
                axs[1].set_ylabel('Mel Bins')
                plt.colorbar(im2, ax=axs[1])
                
                # Denoised
                denoised_mel = denoised[i, 0].cpu().numpy()
                im3 = axs[2].imshow(denoised_mel, origin='lower', aspect='auto', cmap='viridis')
                axs[2].set_title('Denoised Mel Spectrogram')
                axs[2].set_xlabel('Time Frames')
                axs[2].set_ylabel('Mel Bins')
                plt.colorbar(im3, ax=axs[2])
                
                plt.tight_layout()
                
                # Convert the plot to a tensor for logging
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img = Image.open(buf)
                img_tensor = torchvision.transforms.ToTensor()(img)
                
                # Log the image in TensorBoard
                self.logger.experiment.add_image(f'val_sample_{i}', img_tensor, self.global_step_counter)
                
                plt.close(fig)
                
            # Log conditioning information and generate samples
            if conditioning is not None:
                # Take a single conditioning example from each batch item
                for i in range(min(2, original.size(0))):  # Limit to first 2 samples to avoid clutter
                    single_cond = {}
                    for key, value in conditioning.items():
                        if value is not None:
                            single_cond[key] = value[i:i+1]  # Take the i-th batch element
                    
                    # Visualize conditioning information
                    fig = self._visualize_conditioning(single_cond, original.size(3))
                    if fig is not None:
                        # Convert the plot to a tensor for logging
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        img = Image.open(buf)
                        img_tensor = torchvision.transforms.ToTensor()(img)
                        
                        # Log the image in TensorBoard
                        self.logger.experiment.add_image(f'conditioning_{i}', img_tensor, self.global_step_counter)
                        plt.close(fig)
                    
                    # Generate from conditioning
                    freq_bins = self.config['model']['mel_bins']
                    time_frames = original.size(3)
                    
                    # Generate from random noise with conditioning
                    with torch.no_grad():
                        device = original.device
                        seed = torch.randn(1, 1, freq_bins, time_frames, device=device)
                        
                        # Generate
                        generated = self(seed, single_cond)
                        
                        # Log the generation
                        fig, axs = plt.subplots(1, 1, figsize=(10, 4))
                        gen_mel = generated[0, 0].cpu().numpy()
                        im = axs.imshow(gen_mel, origin='lower', aspect='auto', cmap='viridis')
                        axs.set_title(f'Generated Mel Spectrogram (Sample {i})')
                        axs.set_xlabel('Time Frames')
                        axs.set_ylabel('Mel Bins')
                        plt.colorbar(im, ax=axs)
                        
                        plt.tight_layout()
                        
                        # Convert the plot to a tensor for logging
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        buf.seek(0)
                        img = Image.open(buf)
                        img_tensor = torchvision.transforms.ToTensor()(img)
                        
                        # Log the image in TensorBoard
                        self.logger.experiment.add_image(f'generated_sample_{i}', img_tensor, self.global_step_counter)
                        
                        plt.close(fig)
        
        except Exception as e:
            print(f"Error in validation visualization: {e}")
    
    def generate(self, conditioning, max_frames=1000, temperature=1.0):
        """
        Generate a mel spectrogram from conditioning
        
        Args:
            conditioning: Dict containing conditioning features
            max_frames: Maximum number of frames to generate
            temperature: Temperature for generation (higher = more diverse)
            
        Returns:
            Generated mel spectrogram
        """
        device = self.device
        batch_size = 1
        
        # Determine mel shape from conditioning or config
        freq_bins = self.config['model']['mel_bins']
        time_frames = max_frames
        
        # Start with random noise
        mel = torch.randn(batch_size, 1, freq_bins, time_frames, device=device) * temperature
        
        # Simple approach: directly feed noise + conditioning through model 
        # This works for noise conditioning hybrid but not for true diffusion
        with torch.no_grad():
            generated = self(mel, conditioning)
        
        return generated
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Get parameters from config
        learning_rate = self.config['train'].get('learning_rate', 0.001)
        weight_decay = self.config['train'].get('weight_decay', 0.0001)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Create scheduler
        scheduler_type = self.config['train'].get('lr_scheduler', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            lr_patience = self.config['train'].get('lr_patience', 5)
            lr_factor = self.config['train'].get('lr_factor', 0.5)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=lr_factor,
                patience=lr_patience,
                verbose=True
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }