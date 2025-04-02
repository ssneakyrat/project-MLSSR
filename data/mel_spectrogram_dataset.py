import torch
import os
from torch.utils.data import Dataset
import numpy as np
import pytorch_lightning as pl
from utils.utils_transform import normalize_mel_spectrogram, pad_or_truncate_mel
from torch.utils.data import DataLoader, random_split
from utils.utils_datasets import load_dataset

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
                # Store the original shape for debugging
                item['original_shape'] = item['mel_spec'].shape
                self.valid_items.append(item)
        
        if len(self.valid_items) == 0:
            raise ValueError("No valid items with mel spectrograms found in the dataset")
            
        # Print shape statistics
        shapes = [item['original_shape'] for item in self.valid_items]
        print(f"Loaded {len(self.valid_items)} mel spectrograms")
        print(f"Target shape: ({self.target_bins}, {self.target_length})")
        
        # Count how many need padding vs truncation in time dimension
        need_padding = sum(1 for shape in shapes if shape[1] < self.target_length)
        need_truncation = sum(1 for shape in shapes if shape[1] > self.target_length)
        exact_match = sum(1 for shape in shapes if shape[1] == self.target_length)
        
        print(f"Time dimension statistics:")
        print(f"  - Need padding: {need_padding} items (time frames < {self.target_length})")
        print(f"  - Need truncation: {need_truncation} items (time frames > {self.target_length})")
        print(f"  - Exact match: {exact_match} items (time frames = {self.target_length})")
    
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
        
        # Ensure mel_spec is in the right orientation (F, T) where F=frequency bins, T=time frames
        # In our case, we want shape (80, T) where 80 is the number of mel bins
        if mel_spec.shape[0] != self.target_bins:
            # If the first dimension is not the frequency dimension, transpose
            mel_spec = mel_spec.T
            
        # Normalize the mel spectrogram to [0, 1] range
        mel_spec = normalize_mel_spectrogram(mel_spec)
        
        # Pad or truncate to the target dimensions
        # This handles cases where the time dimension doesn't match the expected length
        mel_spec = pad_or_truncate_mel(mel_spec, self.target_length, self.target_bins)
        
        # Convert to tensor and add channel dimension
        # The result will have shape [1, F, T] = [1, 80, 128]
        mel_tensor = torch.from_numpy(mel_spec).float().unsqueeze(0)
        
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

class MelSpectrogramDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for mel spectrogram dataset.
    """
    def __init__(self, config):
        """
        Initialize the data module.
        
        Args:
            config (dict): Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.batch_size = config['train']['batch_size']
        self.num_workers = config['train'].get('num_workers', 4)
        self.pin_memory = config['train'].get('pin_memory', True)
        self.validation_split = config['train'].get('validation_split', 0.1)
        
        # Set dataset parameters
        self.target_length = config['model'].get('time_frames', 128)
        self.target_bins = config['model'].get('mel_bins', 80)
        
        # Track dataset instances
        self.train_dataset = None
        self.val_dataset = None
        
    def prepare_data(self):
        """
        Download or prepare data if necessary.
        This method is called only once and on only one GPU.
        """
        # Nothing to do here as data is already prepared
        pass
        
    def setup(self, stage=None):
        """
        Setup train/val/test datasets.
        This method is called on every GPU.
        
        Args:
            stage (str, optional): 'fit', 'validate', 'test', or 'predict'
        """
        # Load all data items
        data_items = load_dataset(split='train', shuffle=True)
        
        if len(data_items) == 0:
            raise ValueError("No data items found. Make sure the dataset is properly prepared.")
        
        # Filter items to only include those with mel spectrograms
        valid_items = []
        for item in data_items:
            if 'mel_spec' in item and item['mel_spec'] is not None:
                valid_items.append(item)
        
        if len(valid_items) == 0:
            raise ValueError("No valid items with mel spectrograms found in the dataset. Check preprocessing.")
            
        print(f"Found {len(valid_items)} valid items with mel spectrograms out of {len(data_items)} total items.")
        
        # Split into train and validation sets
        val_size = int(len(valid_items) * self.validation_split)
        train_size = len(valid_items) - val_size
        
        train_items, val_items = random_split(
            valid_items, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create dataset instances
        self.train_dataset = MelSpectrogramDataset(
            list(train_items), 
            target_length=self.target_length, 
            target_bins=self.target_bins
        )
        
        self.val_dataset = MelSpectrogramDataset(
            list(val_items), 
            target_length=self.target_length, 
            target_bins=self.target_bins
        )
        
        print(f"Setup complete. Train dataset: {len(self.train_dataset)} items, "
              f"Validation dataset: {len(self.val_dataset)} items")
        
    def train_dataloader(self):
        """
        Create the training dataloader.
        
        Returns:
            DataLoader: Training dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=collate_fn
        )
        
    def val_dataloader(self):
        """
        Create the validation dataloader.
        
        Returns:
            DataLoader: Validation dataloader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn
        )