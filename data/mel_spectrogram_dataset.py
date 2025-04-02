import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from utils.utils_datasets import load_dataset
from data.mel_spectrogram_dataset import MelSpectrogramDataset, collate_fn


class MelSpectrogramDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Mel Spectrogram reconstruction task.
    """
    def __init__(self, config):
        """
        Initialize the DataModule.
        
        Args:
            config (dict): Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.batch_size = config['train'].get('batch_size', 32)
        self.num_workers = config['train'].get('num_workers', 4)
        self.pin_memory = config['train'].get('pin_memory', True)
        self.validation_split = config['train'].get('validation_split', 0.1)
        
        # For storing dataset splits
        self.train_dataset = None
        self.val_dataset = None
        self.data_items = None
    
    def prepare_data(self):
        """
        Perform operations that should be done only once and 
        can be done in a non-distributed setting (e.g., download data).
        
        Note: This method is called only once and on only one GPU.
        """
        # Nothing to do here since we assume data is already preprocessed
        pass
    
    def setup(self, stage=None):
        """
        Perform operations to setup the data for training/validation.
        
        Args:
            stage (str): Either 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'fit' or stage is None:
            # Load dataset
            print("Loading dataset...")
            self.data_items = load_dataset(split='train', shuffle=True)
            print(f"Loaded {len(self.data_items)} samples")
            
            # Check if at least one sample contains mel spectrogram
            has_mel = any('mel_spec' in item for item in self.data_items)
            if not has_mel:
                raise ValueError("Error: No mel spectrograms found in the dataset")
            
            # Split data into training and validation sets
            num_val = int(len(self.data_items) * self.validation_split)
            num_train = len(self.data_items) - num_val
            
            # Create indices for random splitting
            indices = list(range(len(self.data_items)))
            np.random.shuffle(indices)
            
            train_indices = indices[num_val:]
            val_indices = indices[:num_val]
            
            train_data = [self.data_items[i] for i in train_indices]
            val_data = [self.data_items[i] for i in val_indices]
            
            # Create datasets
            self.train_dataset = MelSpectrogramDataset(
                train_data,
                target_length=self.config['model'].get('time_frames', 128),
                target_bins=self.config['model'].get('mel_bins', 80)
            )
            
            self.val_dataset = MelSpectrogramDataset(
                val_data,
                target_length=self.config['model'].get('time_frames', 128),
                target_bins=self.config['model'].get('mel_bins', 80)
            )
            
            print(f"Training samples: {len(self.train_dataset)}")
            print(f"Validation samples: {len(self.val_dataset)}")
    
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
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
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
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )