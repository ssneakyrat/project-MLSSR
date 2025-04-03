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
    
    # Modified calculate_adversarial_loss function in models/hifi_losses.py
    def calculate_adversarial_loss(self, disc_outputs_real, disc_outputs_fake, is_generator=False):
        """Calculate adversarial loss using hinge loss with improved numeric stability
        
        Args:
            disc_outputs_real: List of discrimination scores for real data
            disc_outputs_fake: List of discrimination scores for generated data
            is_generator: If True, calculate generator loss; otherwise, discriminator loss
            
        Returns:
            loss: Adversarial loss value
        """
        # Convert inputs to lists if they're not already
        if disc_outputs_fake is None:
            # Handle case where no discriminator outputs were provided
            # Return a valid tensor to avoid breaking the training
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        if not isinstance(disc_outputs_fake, list):
            disc_outputs_fake = [disc_outputs_fake]
        
        if disc_outputs_real is not None and not isinstance(disc_outputs_real, list):
            disc_outputs_real = [disc_outputs_real]
        
        if is_generator:
            # Generator tries to maximize discriminator outputs for fake data
            loss = 0
            valid_outputs = 0
            
            for d_fake in disc_outputs_fake:
                # Skip if tensor contains NaN values
                if d_fake is None or torch.isnan(d_fake).any():
                    continue
                
                # Clamp the discriminator outputs to prevent extreme values
                d_fake_clamped = torch.clamp(d_fake, min=-20.0, max=20.0)
                
                # Use the standard generator loss: E[(1 - D(G(z)))²]
                batch_loss = torch.mean((1 - d_fake_clamped) ** 2)
                
                # Skip if the loss is NaN
                if torch.isnan(batch_loss).any():
                    continue
                
                loss += batch_loss
                valid_outputs += 1
            
            if valid_outputs > 0:
                return loss / valid_outputs
            else:
                print("Warning: No valid outputs for generator adversarial loss")
                # Return a small positive value to avoid breaking training
                return torch.tensor(0.1, device=self.device, requires_grad=True)
        else:
            # Discriminator tries to maximize output for real and minimize for fake
            if disc_outputs_real is None:
                print("Warning: No real outputs for discriminator adversarial loss")
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            
            loss_real = 0
            loss_fake = 0
            valid_pairs = 0
            
            for d_real, d_fake in zip(disc_outputs_real, disc_outputs_fake):
                # Skip if either tensor contains NaN values
                if d_real is None or d_fake is None or torch.isnan(d_real).any() or torch.isnan(d_fake).any():
                    continue
                
                # Clamp the discriminator outputs to prevent extreme values
                d_real_clamped = torch.clamp(d_real, min=-20.0, max=20.0)
                d_fake_clamped = torch.clamp(d_fake, min=-20.0, max=20.0)
                
                # Calculate discriminator loss components
                batch_loss_real = torch.mean((1 - d_real_clamped) ** 2)
                batch_loss_fake = torch.mean(d_fake_clamped ** 2)
                
                # Skip if either loss is NaN
                if torch.isnan(batch_loss_real).any() or torch.isnan(batch_loss_fake).any():
                    continue
                
                loss_real += batch_loss_real
                loss_fake += batch_loss_fake
                valid_pairs += 1
            
            if valid_pairs > 0:
                return (loss_real + loss_fake) / valid_pairs
            else:
                print("Warning: No valid output pairs for discriminator adversarial loss")
                # Return a small positive value to avoid breaking training
                return torch.tensor(0.1, device=self.device, requires_grad=True)
    
    # Modified calculate_feature_matching_loss function in models/hifi_losses.py
    def calculate_feature_matching_loss(self, feat_maps_real, feat_maps_fake):
        """Calculate feature matching loss with improved error handling
        
        Args:
            feat_maps_real: List of feature maps from discriminator for real audio
            feat_maps_fake: List of feature maps from discriminator for generated audio
            
        Returns:
            loss: Feature matching loss value
        """
        # Handle empty inputs
        if not feat_maps_real or not feat_maps_fake:
            return torch.tensor(0.0, device=self.device)
        
        # Convert to list format if needed
        if not isinstance(feat_maps_real[0], list):
            feat_maps_real = [feat_maps_real]
            feat_maps_fake = [feat_maps_fake]
        
        # Check for mismatch in outer list length
        if len(feat_maps_real) != len(feat_maps_fake):
            print(f"Warning: Mismatch in feature maps list lengths. Real: {len(feat_maps_real)}, Fake: {len(feat_maps_fake)}")
            min_len = min(len(feat_maps_real), len(feat_maps_fake))
            feat_maps_real = feat_maps_real[:min_len]
            feat_maps_fake = feat_maps_fake[:min_len]
        
        loss = 0
        total_layers = 0
        valid_pairs = 0
        
        # Iterate through discriminators
        for disc_idx, (fmaps_real_d, fmaps_fake_d) in enumerate(zip(feat_maps_real, feat_maps_fake)):
            # Check for mismatch in inner list length
            if len(fmaps_real_d) != len(fmaps_fake_d):
                print(f"Warning: Mismatch in feature maps count for discriminator {disc_idx}. Real: {len(fmaps_real_d)}, Fake: {len(fmaps_fake_d)}")
                min_len = min(len(fmaps_real_d), len(fmaps_fake_d))
                fmaps_real_d = fmaps_real_d[:min_len]
                fmaps_fake_d = fmaps_fake_d[:min_len]
            
            # Iterate through feature maps from each layer
            for layer_idx, (fmap_r, fmap_f) in enumerate(zip(fmaps_real_d, fmaps_fake_d)):
                try:
                    # Skip if either tensor has NaN values
                    if torch.isnan(fmap_r).any() or torch.isnan(fmap_f).any():
                        print(f"Warning: NaN found in feature maps at disc_idx={disc_idx}, layer_idx={layer_idx}")
                        continue
                    
                    # Skip if dimensions don't match and can't be easily fixed
                    if fmap_r.dim() != fmap_f.dim():
                        print(f"Warning: Dimension mismatch in feature maps at disc_idx={disc_idx}, layer_idx={layer_idx}")
                        continue
                    
                    # Handle different feature map sizes by cropping to the smaller size for each dimension > 1
                    # (first two dimensions are batch_size and channels)
                    matched_tensors = [fmap_r, fmap_f]
                    for dim in range(2, fmap_r.dim()):
                        sizes = [t.size(dim) for t in matched_tensors]
                        min_size = min(sizes)
                        
                        # Crop each tensor to match the minimum size in this dimension
                        for i in range(len(matched_tensors)):
                            if matched_tensors[i].size(dim) > min_size:
                                # Create slice for this dimension
                                slices = [slice(None)] * matched_tensors[i].dim()
                                slices[dim] = slice(0, min_size)
                                matched_tensors[i] = matched_tensors[i][slices]
                    
                    # Retrieve matched tensors
                    fmap_r_matched, fmap_f_matched = matched_tensors
                    
                    # Calculate L1 loss on the matched size
                    layer_loss = F.l1_loss(fmap_f_matched, fmap_r_matched.detach())
                    
                    # Check for NaN in the loss and skip if found
                    if torch.isnan(layer_loss).any():
                        print(f"Warning: NaN in L1 loss at disc_idx={disc_idx}, layer_idx={layer_idx}")
                        continue
                    
                    # Add to total loss
                    loss = loss + layer_loss
                    valid_pairs += 1
                    
                except Exception as e:
                    print(f"Error processing feature maps at disc_idx={disc_idx}, layer_idx={layer_idx}: {e}")
                    continue
                
                total_layers += 1
        
        # Normalize by number of valid feature map pairs
        if valid_pairs > 0:
            return loss / valid_pairs
        else:
            print("Warning: No valid feature map pairs found for feature matching loss")
            # Return zero loss but keep gradient tracking
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def calculate_mel_loss(self, audio_real, audio_fake):
        """Calculate mel-spectrogram reconstruction loss with length matching
        
        Args:
            audio_real: Real waveform [B, 1, T]
            audio_fake: Generated waveform [B, 1, T]
            
        Returns:
            loss: L1 loss between real and fake mel-spectrograms
        """
        # Handle different audio lengths by cropping to the smaller size
        if audio_real.size(2) != audio_fake.size(2):
            min_size = min(audio_real.size(2), audio_fake.size(2))
            audio_real = audio_real[:, :, :min_size]
            audio_fake = audio_fake[:, :, :min_size]
        
        # Extract mel-spectrograms from audio
        mel_real = self.audio_to_mel(audio_real.squeeze(1))
        mel_fake = self.audio_to_mel(audio_fake.squeeze(1))
        
        # L1 loss between mel-spectrograms
        mel_loss = F.l1_loss(mel_fake, mel_real)
        
        return mel_loss
    
    # Modified audio_to_mel function in models/hifi_losses.py
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
        
        # Add small epsilon to avoid zero input which can cause NaN in STFT
        audio = audio + 1e-6 * torch.randn_like(audio)
        
        # Compute STFT with error handling
        try:
            stft = torch.stft(
                audio,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=torch.hann_window(win_length).to(audio.device),
                return_complex=True
            )
            
            # Convert to power spectrogram with numeric stability fixes
            spec = torch.abs(stft) ** 2
            spec = torch.clamp(spec, min=1e-9)  # Prevent zeros
            
            # Create mel filterbank (cached to improve performance)
            if not hasattr(self, 'mel_basis') or self.mel_basis.device != audio.device:
                self.mel_basis = self._create_mel_filterbank(
                    n_fft=n_fft,
                    n_mels=n_mels,
                    sample_rate=sample_rate,
                    fmin=fmin,
                    fmax=fmax
                ).to(audio.device)
            
            # Apply mel filterbank with numeric stability
            mel = torch.matmul(self.mel_basis, spec)
            mel = torch.clamp(mel, min=1e-9)  # Ensure strictly positive before log
            
            # Convert to log scale with safer calculation
            log_mel = torch.log10(mel)
            
            # Check for NaN and replace with small values if needed
            if torch.isnan(log_mel).any():
                log_mel = torch.where(torch.isnan(log_mel), 
                                    torch.tensor(-5.0, device=log_mel.device), 
                                    log_mel)
            
            return log_mel
        
        except Exception as e:
            print(f"Error in audio_to_mel: {e}")
            # Return a valid tensor to avoid breaking the training
            return torch.zeros((audio.shape[0], n_mels, audio.shape[1] // hop_length + 1), 
                            device=audio.device) - 5.0  # -5.0 is a reasonable log10 minimum value
    
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