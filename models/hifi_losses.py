import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np


class HiFiGANLoss:
    """Combined loss functions for HiFi-GAN training"""
    
    def __init__(self, 
                 lambda_adv=1.0,      # Weight for adversarial loss
                 lambda_fm=2.0,       # Weight for feature matching loss
                 lambda_mel=45.0,     # Weight for mel-spectrogram loss
                 mel_config=None      # Config for mel-spectrogram extraction
                ):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        
        # Configure mel-spectrogram settings for loss calculation
        self.mel_config = mel_config if mel_config is not None else {
            'sample_rate': 22050,
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'n_mels': 80,
            'fmin': 0,
            'fmax': 8000
        }
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def calculate_adversarial_loss(self, disc_outputs_real, disc_outputs_fake, is_generator=False):
        """Calculate adversarial loss using hinge loss
        
        Args:
            disc_outputs_real: List of discrimination scores for real data
            disc_outputs_fake: List of discrimination scores for generated data
            is_generator: If True, calculate generator loss; otherwise, discriminator loss
            
        Returns:
            loss: Adversarial loss value
        """
        # Convert inputs to lists if they're not already
        if not isinstance(disc_outputs_real, list):
            disc_outputs_real = [disc_outputs_real]
        if not isinstance(disc_outputs_fake, list):
            disc_outputs_fake = [disc_outputs_fake]
        
        if is_generator:
            # Generator tries to maximize discriminator outputs for fake data
            loss = 0
            for d_fake in disc_outputs_fake:
                loss += torch.mean((1 - d_fake) ** 2)
            return loss / len(disc_outputs_fake)
        else:
            # Discriminator tries to maximize output for real and minimize for fake
            loss_real = 0
            loss_fake = 0
            for d_real, d_fake in zip(disc_outputs_real, disc_outputs_fake):
                loss_real += torch.mean((1 - d_real) ** 2)
                loss_fake += torch.mean(d_fake ** 2)
            return (loss_real + loss_fake) / len(disc_outputs_real)
    
    def calculate_feature_matching_loss(self, feat_maps_real, feat_maps_fake):
        """Calculate feature matching loss
        
        Args:
            feat_maps_real: List of feature maps from discriminator for real audio
            feat_maps_fake: List of feature maps from discriminator for generated audio
            
        Returns:
            loss: Feature matching loss value
        """
        if not isinstance(feat_maps_real[0], list):
            feat_maps_real = [feat_maps_real]
            feat_maps_fake = [feat_maps_fake]
        
        loss = 0
        
        # Iterate through discriminators
        for fmaps_real_d, fmaps_fake_d in zip(feat_maps_real, feat_maps_fake):
            # Iterate through feature maps from each layer
            for fmap_r, fmap_f in zip(fmaps_real_d, fmaps_fake_d):
                loss += F.l1_loss(fmap_f, fmap_r.detach())
        
        # Normalize by number of feature maps
        total_layers = sum(len(fm) for fm in feat_maps_real)
        return loss / total_layers
    
    def calculate_mel_loss(self, audio_real, audio_fake):
        """Calculate mel-spectrogram reconstruction loss
        
        Args:
            audio_real: Real waveform [B, 1, T]
            audio_fake: Generated waveform [B, 1, T]
            
        Returns:
            loss: L1 loss between real and fake mel-spectrograms
        """
        # Extract mel-spectrograms from audio
        mel_real = self.audio_to_mel(audio_real.squeeze(1))
        mel_fake = self.audio_to_mel(audio_fake.squeeze(1))
        
        # L1 loss between mel-spectrograms
        mel_loss = F.l1_loss(mel_fake, mel_real)
        
        return mel_loss
    
    def audio_to_mel(self, audio):
        """Convert audio waveform to mel-spectrogram using torch operations
        
        Args:
            audio: Audio waveform [B, T]
            
        Returns:
            mel: Mel-spectrogram [B, n_mels, T']
        """
        # Extract config parameters
        n_fft = self.mel_config['n_fft']
        hop_length = self.mel_config['hop_length']
        win_length = self.mel_config['win_length']
        n_mels = self.mel_config['n_mels']
        fmin = self.mel_config['fmin']
        fmax = self.mel_config['fmax']
        sample_rate = self.mel_config['sample_rate']
        
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(audio.device),
            return_complex=True
        )
        
        # Convert to power spectrogram
        spec = torch.abs(stft) ** 2
        
        # Create mel filterbank (cached to improve performance)
        if not hasattr(self, 'mel_basis') or self.mel_basis.device != audio.device:
            self.mel_basis = self._create_mel_filterbank(
                n_fft=n_fft,
                n_mels=n_mels,
                sample_rate=sample_rate,
                fmin=fmin,
                fmax=fmax
            ).to(audio.device)
        
        # Apply mel filterbank
        mel = torch.matmul(self.mel_basis, spec)
        
        # Convert to log scale
        log_mel = torch.log10(torch.clamp(mel, min=1e-5))
        
        return log_mel
    
    def _create_mel_filterbank(self, n_fft, n_mels, sample_rate, fmin, fmax):
        """Create a mel filterbank matrix using the same approach as librosa
        
        Args:
            n_fft: FFT window size
            n_mels: Number of mel bands
            sample_rate: Audio sample rate
            fmin: Minimum frequency
            fmax: Maximum frequency
            
        Returns:
            mel_basis: Mel filterbank matrix
        """
        # Create a filterbank matrix like librosa.filters.mel
        # We only need to compute this once and then cache it
        n_freqs = n_fft // 2 + 1
        
        # Convert Hz to mel
        min_mel = 2595 * np.log10(1 + fmin / 700)
        max_mel = 2595 * np.log10(1 + fmax / 700)
        
        # Create mel frequency array
        mels = torch.linspace(min_mel, max_mel, n_mels + 2)
        
        # Convert mel back to Hz
        freqs = 700 * (10 ** (mels / 2595) - 1)
        
        # Convert Hz to FFT bins
        bins = torch.floor((n_fft + 1) * freqs / sample_rate).long()
        
        # Create filterbank
        fb = torch.zeros(n_mels, n_freqs)
        
        for i in range(n_mels):
            # Lower and upper frequency bins
            fmin_bin = bins[i]
            fcenter_bin = bins[i + 1]
            fmax_bin = bins[i + 2]
            
            # Create triangular filter
            for j in range(fmin_bin, fcenter_bin + 1):
                if j < n_freqs:
                    fb[i, j] = (j - fmin_bin) / (fcenter_bin - fmin_bin)
            
            for j in range(fcenter_bin, fmax_bin + 1):
                if j < n_freqs:
                    fb[i, j] = (fmax_bin - j) / (fmax_bin - fcenter_bin)
        
        return fb
    
    def generator_loss(self, disc_outputs_fake, feat_maps_real, feat_maps_fake, audio_real, audio_fake):
        """Calculate generator loss (adversarial + feature matching + mel-spectrogram)
        
        Args:
            disc_outputs_fake: Discrimination scores for generated audio
            feat_maps_real: Feature maps from discriminator for real audio
            feat_maps_fake: Feature maps from discriminator for generated audio
            audio_real: Real waveform
            audio_fake: Generated waveform
            
        Returns:
            total_loss: Combined generator loss
            losses: Dictionary with individual loss components
        """
        # Adversarial loss
        adv_loss = self.calculate_adversarial_loss(
            None, disc_outputs_fake, is_generator=True
        )
        
        # Feature matching loss
        fm_loss = self.calculate_feature_matching_loss(
            feat_maps_real, feat_maps_fake
        )
        
        # Mel-spectrogram reconstruction loss
        mel_loss = self.calculate_mel_loss(audio_real, audio_fake)
        
        # Combine losses with weights
        total_loss = (
            self.lambda_adv * adv_loss +
            self.lambda_fm * fm_loss +
            self.lambda_mel * mel_loss
        )
        
        # Store individual losses for logging
        losses = {
            'g_total': total_loss.item(),
            'g_adv': adv_loss.item(),
            'g_fm': fm_loss.item(),
            'g_mel': mel_loss.item()
        }
        
        return total_loss, losses
    
    def discriminator_loss(self, disc_outputs_real, disc_outputs_fake):
        """Calculate discriminator loss
        
        Args:
            disc_outputs_real: Discrimination scores for real audio
            disc_outputs_fake: Discrimination scores for generated audio
            
        Returns:
            loss: Discriminator loss
            loss_dict: Dictionary with individual loss components
        """
        # Adversarial loss
        d_loss = self.calculate_adversarial_loss(
            disc_outputs_real, disc_outputs_fake, is_generator=False
        )
        
        loss_dict = {'d_loss': d_loss.item()}
        
        return d_loss, loss_dict