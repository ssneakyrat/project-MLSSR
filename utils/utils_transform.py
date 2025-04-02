import torch
import numpy as np

def normalize_mel_spectrogram(mel_spec):
    """
    Normalize mel spectrogram to range [0, 1].
    
    Args:
        mel_spec (numpy.ndarray): Mel spectrogram in dB scale
        
    Returns:
        numpy.ndarray: Normalized mel spectrogram
    """
    # Convert from dB scale if necessary (librosa.power_to_db output is typically -80 to 0)
    if np.min(mel_spec) < 0:
        # Assuming the mel_spec is already in dB scale, normalize from typical dB range
        # Clip to reasonable dB range to avoid outliers
        mel_spec = np.clip(mel_spec, -80.0, 0.0)
        # Normalize to [0, 1]
        mel_spec = (mel_spec + 80.0) / 80.0
    else:
        # If already positive, just normalize to [0, 1]
        max_val = np.max(mel_spec)
        min_val = np.min(mel_spec)
        if max_val > min_val:  # Avoid division by zero
            mel_spec = (mel_spec - min_val) / (max_val - min_val)
    
    return mel_spec

def pad_or_truncate_mel(mel_spec, target_length=128, target_bins=80):
    """
    Resize mel spectrogram to target dimensions by padding or truncating.
    
    Args:
        mel_spec (numpy.ndarray): Mel spectrogram of shape [F, T]
                                  where F is num_mels and T is time frames
        target_length (int): Target number of time frames
        target_bins (int): Target number of mel bins
        
    Returns:
        numpy.ndarray: Resized mel spectrogram of shape [target_bins, target_length]
    """
    # Get current dimensions
    curr_bins, curr_length = mel_spec.shape
    
    # Prepare output array (initialized with zeros for padding)
    resized_mel = np.zeros((target_bins, target_length), dtype=mel_spec.dtype)
    
    # Copy mel bins (frequency dimension)
    freq_bins_to_copy = min(curr_bins, target_bins)
    
    # Handle time dimension
    time_frames_to_copy = min(curr_length, target_length)
    
    # Copy the content from the original mel spectrogram to the resized one
    # This fixes the bug in the original implementation where it was overwriting itself
    resized_mel[:freq_bins_to_copy, :time_frames_to_copy] = mel_spec[:freq_bins_to_copy, :time_frames_to_copy]
    
    return resized_mel

def prepare_mel_for_model(mel_spec, target_length=128, target_bins=80):
    """
    Prepare mel spectrogram for the model:
    1. Normalize to [0, 1]
    2. Pad or truncate to target dimensions
    3. Add channel dimension and convert to proper orientation for U-Net
    
    Args:
        mel_spec (numpy.ndarray): Mel spectrogram
        target_length (int): Target number of time frames
        target_bins (int): Target number of mel bins
        
    Returns:
        torch.Tensor: Processed mel spectrogram of shape [1, target_length, target_bins]
    """
    # Ensure mel_spec is in the right orientation (F, T)
    if mel_spec.shape[0] > mel_spec.shape[1]:
        # If frequency dimension is larger than time dimension, transpose
        mel_spec = mel_spec.T
    
    # Normalize
    mel_spec = normalize_mel_spectrogram(mel_spec)
    
    # Pad or truncate
    mel_spec = pad_or_truncate_mel(mel_spec, target_length, target_bins)
    
    # Convert to tensor and add channel dimension
    mel_tensor = torch.from_numpy(mel_spec).float()
    
    # Reshape to (1, T, F) - add channel dimension and ensure time is before frequency
    mel_tensor = mel_tensor.permute(1, 0).unsqueeze(0)
    
    return mel_tensor

def batch_prepare_mel_spectrograms(mel_specs, target_length=128, target_bins=80):
    """
    Process a batch of mel spectrograms for the model.
    
    Args:
        mel_specs (list): List of mel spectrograms
        target_length (int): Target number of time frames
        target_bins (int): Target number of mel bins
        
    Returns:
        torch.Tensor: Batch of processed mel spectrograms of shape [B, 1, target_length, target_bins]
    """
    processed_mels = []
    
    for mel_spec in mel_specs:
        # Process each spectrogram
        processed_mel = prepare_mel_for_model(mel_spec, target_length, target_bins)
        processed_mels.append(processed_mel)
    
    # Stack into batch
    if processed_mels:
        return torch.stack(processed_mels, dim=0)
    else:
        return torch.zeros((0, 1, target_length, target_bins))