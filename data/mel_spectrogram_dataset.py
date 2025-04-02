import torch
from torch.utils.data import Dataset
import numpy as np

from utils.utils_transform import normalize_mel_spectrogram, pad_or_truncate_mel

class MelSpectrogramDataset(Dataset):
    """
    Dataset for mel spectrogram reconstruction.
    """
    def __init__(self, data_items, target_length=128, target_bins=80):
        """
        Initialize the dataset.
        
        Args:
            data_items (list): List of data items from load_dataset()
            target_length (int): Target number of time frames
            target_bins (int): Target number of mel bins
        """
        self.data_items = data_items
        self.target_length = target_length
        self.target_bins = target_bins
        
        # Filter out items that don't have mel spectrograms
        self.valid_items = []
        for item in data_items:
            if 'mel_spec' in item and item['mel_spec'] is not None:
                self.valid_items.append(item)
        
        if len(self.valid_items) == 0:
            raise ValueError("No valid items with mel spectrograms found in the dataset")
    
    def __len__(self):
        """
        Get the number of items in the dataset.
        
        Returns:
            int: Number of items
        """
        return len(self.valid_items)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            torch.Tensor: Processed mel spectrogram tensor of shape [1, T, F]
                         where T is target_length and F is target_bins
        """
        # Get the data item
        item = self.valid_items[idx]
        
        # Get mel spectrogram
        mel_spec = item['mel_spec']
        
        # Ensure mel_spec is in the right orientation (F, T)
        if mel_spec.shape[0] < mel_spec.shape[1]:
            # If time is second dimension (longer), no need to transpose
            pass
        else:
            # If time is first dimension, transpose
            mel_spec = mel_spec.T
        
        # Normalize
        mel_spec = normalize_mel_spectrogram(mel_spec)
        
        # Pad or truncate
        mel_spec = pad_or_truncate_mel(mel_spec, self.target_length, self.target_bins)
        
        # Convert to tensor and add channel dimension
        mel_tensor = torch.from_numpy(mel_spec).float()
        
        # Reshape to [1, F, T] for 2D convolution (batch, channels, height, width)
        mel_tensor = mel_tensor.unsqueeze(0)
        
        return mel_tensor

def collate_fn(batch):
    """
    Custom collate function.
    
    Args:
        batch (list): List of tensors
        
    Returns:
        torch.Tensor: Batch tensor
    """
    # Simply stack tensors along the batch dimension
    return torch.stack(batch, dim=0)