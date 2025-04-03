import torch
import torch.nn as nn
import torch.nn.functional as F

class PhonemeEncoder(nn.Module):
    """Encodes phoneme sequences into feature representations"""
    def __init__(self, num_phonemes, embedding_dim=256, hidden_dim=256):
        super().__init__()
        self.phoneme_embedding = nn.Embedding(num_phonemes, embedding_dim)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        ])
        self.norm_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        ])
        
    def forward(self, phoneme_ids, phoneme_durations=None):
        """
        Args:
            phoneme_ids: Tensor of shape [batch_size, max_phonemes]
            phoneme_durations: Tensor of shape [batch_size, max_phonemes]
                               representing the duration of each phoneme in frames
        Returns:
            Tensor of shape [batch_size, hidden_dim, max_frames]
        """
        # Embed phonemes
        x = self.phoneme_embedding(phoneme_ids)  # [B, P, E]
        x = x.transpose(1, 2)  # [B, E, P]
        
        # Apply convolutional layers
        for conv, norm in zip(self.conv_layers, self.norm_layers):
            x = F.relu(norm(conv(x)))
        
        # If durations are provided, expand phoneme features to frame level
        if phoneme_durations is not None:
            # Convert phoneme-level features to frame-level
            frame_level_features = []
            
            batch_size = phoneme_ids.size(0)
            max_frames = int(phoneme_durations.sum(dim=1).max().item())
            
            for b in range(batch_size):
                # Initialize output tensor
                frames_features = torch.zeros(
                    x.size(1), max_frames, device=x.device
                )
                
                # Fill in frame-level features
                start_frame = 0
                for p, duration in enumerate(phoneme_durations[b]):
                    if duration <= 0:
                        continue
                    
                    end_frame = start_frame + int(duration.item())
                    if end_frame > max_frames:
                        end_frame = max_frames
                    
                    # Check if phoneme index is in range
                    if p < x.size(2):
                        # Repeat phoneme features for its duration
                        frames_features[:, start_frame:end_frame] = x[b, :, p].unsqueeze(-1).repeat(1, end_frame - start_frame)
                    
                    start_frame = end_frame
                    if start_frame >= max_frames:
                        break
                
                frame_level_features.append(frames_features)
            
            x = torch.stack(frame_level_features, dim=0)  # [B, H, F]
        
        return x


class PitchEncoder(nn.Module):
    """Encodes pitch (MIDI or F0) information into feature representations"""
    def __init__(self, input_dim=1, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        
        # For MIDI pitch (categorical)
        self.pitch_embedding = nn.Embedding(128, hidden_dim)  # 128 MIDI notes
        
        # For F0 (continuous)
        self.f0_encoder = nn.Sequential(
            nn.Conv1d(1, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Combine pitch and F0
        self.combiner = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, midi_pitch=None, f0=None):
        """
        Args:
            midi_pitch: Tensor of shape [batch_size, max_frames] with MIDI pitch values
            f0: Tensor of shape [batch_size, max_frames] with F0 values
        Returns:
            Tensor of shape [batch_size, hidden_dim, max_frames]
        """
        features = []
        
        # Process MIDI pitch if provided
        if midi_pitch is not None:
            # Clamp MIDI values to valid range
            midi_pitch = torch.clamp(midi_pitch, 0, 127).long()
            pitch_features = self.pitch_embedding(midi_pitch)  # [B, F, H]
            pitch_features = pitch_features.transpose(1, 2)  # [B, H, F]
            features.append(pitch_features)
        
        # Process F0 if provided
        if f0 is not None:
            # Normalize and handle unvoiced frames (F0 = 0)
            f0_normalized = f0 / 500.0  # Normalize to reasonable range
            f0_normalized = torch.where(f0 > 0, f0_normalized, torch.zeros_like(f0_normalized))
            
            f0_features = self.f0_encoder(f0_normalized.unsqueeze(1))  # [B, H, F]
            features.append(f0_features)
        
        # Combine features if both are provided
        if len(features) == 2:
            return self.combiner(torch.cat(features, dim=1))
        elif len(features) == 1:
            return features[0]
        else:
            raise ValueError("At least one of midi_pitch or f0 must be provided")


class ConditioningEncoder(nn.Module):
    """Encodes all conditioning signals into a single representation"""
    def __init__(self, num_phonemes, embedding_dim=256, hidden_dim=256):
        super().__init__()
        
        # Individual encoders
        self.phoneme_encoder = PhonemeEncoder(num_phonemes, embedding_dim, hidden_dim)
        self.pitch_encoder = PitchEncoder(input_dim=1, hidden_dim=hidden_dim)
        
        # Combine all features
        self.combiner = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Project to 2D feature maps for UNet
        self.freq_projection = nn.Conv1d(
            hidden_dim, hidden_dim * 2, kernel_size=1
        )
        
    def forward(self, phoneme_ids=None, phoneme_durations=None, 
            midi_pitch=None, f0=None, mel_shape=None):
        """
        Args:
            phoneme_ids: Tensor of shape [batch_size, max_phonemes]
            phoneme_durations: Tensor of shape [batch_size, max_phonemes] 
            midi_pitch: Tensor of shape [batch_size, max_frames]
            f0: Tensor of shape [batch_size, max_frames]
            mel_shape: Tuple (freq_bins, time_frames) for reshaping
        Returns:
            2D conditioning tensor of shape [batch_size, channels, freq_bins, time_frames]
        """
        features = []
        
        # Process phonemes if provided
        if phoneme_ids is not None:
            phoneme_features = self.phoneme_encoder(phoneme_ids, phoneme_durations)
            features.append(phoneme_features)
        
        # Process pitch information if provided
        if midi_pitch is not None or f0 is not None:
            pitch_features = self.pitch_encoder(midi_pitch, f0)
            features.append(pitch_features)
        
        # Combine all features with alignment handling
        if len(features) > 1:
            # First ensure all features have the same time dimension
            target_length = max(f.shape[2] for f in features)
            
            # Align features to have the same time dimension
            aligned_features = []
            for f in features:
                if f.shape[2] != target_length:
                    # Use interpolation to match the target length
                    aligned_f = F.interpolate(
                        f, 
                        size=target_length, 
                        mode='linear', 
                        align_corners=False
                    )
                    aligned_features.append(aligned_f)
                else:
                    aligned_features.append(f)
            
            # Now combine the aligned features
            x = self.combiner(torch.cat(aligned_features, dim=1))
        elif len(features) == 1:
            x = features[0]
        else:
            # Handle the case where no features are available
            if mel_shape is not None:
                # Create empty features matching the target shape
                freq_bins, time_frames = mel_shape
                batch_size = 1  # Default batch size when no features
                x = torch.zeros(batch_size, self.hidden_dim, time_frames, device=phoneme_ids.device if phoneme_ids is not None else 'cpu')
            else:
                raise ValueError("No conditioning features provided and no mel_shape specified")
        
        # If no mel_shape is provided, return 1D features
        if mel_shape is None:
            return x
        
        # Project to 2D feature maps
        freq_bins, time_frames = mel_shape
        batch_size = x.size(0)
        hidden_dim = x.size(1)
        
        # Make sure our features match the target time frames
        if x.shape[2] != time_frames:
            x = F.interpolate(
                x, 
                size=time_frames, 
                mode='linear', 
                align_corners=False
            )
        
        # Use frequency projection to get features for each frequency bin
        freq_features = self.freq_projection(x)  # [B, H*2, F]
        
        # Reshape to match expected UNet input
        freq_features = freq_features.view(batch_size, 2, hidden_dim, time_frames)
        freq_features = freq_features.permute(0, 2, 1, 3)  # [B, H, 2, F]
        
        # Interpolate in frequency dimension to match mel spectrogram
        freq_features = F.interpolate(
            freq_features, size=(freq_bins, time_frames), 
            mode='bilinear', align_corners=False
        )
        
        return freq_features