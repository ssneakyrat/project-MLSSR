import torch
import os
from torch.utils.data import Dataset
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import h5py

class H5FileManager:
    """
    Singleton class to manage H5 file handles.
    """
    _instance = None
    
    @staticmethod
    def get_instance():
        if H5FileManager._instance is None:
            H5FileManager._instance = H5FileManager()
        return H5FileManager._instance
    
    def __init__(self):
        self.h5_files = {}
    
    def get_file(self, file_path):
        if file_path not in self.h5_files:
            self.h5_files[file_path] = h5py.File(file_path, 'r')
        return self.h5_files[file_path]
    
    def close_all(self):
        for file in self.h5_files.values():
            file.close()
        self.h5_files = {}

class MelSpectrogramDataset(Dataset):
    """
    Dataset for mel spectrogram reconstruction using generic H5 format.
    """
    def __init__(self, h5_path, data_key='mel_spectrograms', lazy_load=True):
        """
        Initialize the dataset.
        
        Args:
            h5_path (str): Path to the H5 file
            data_key (str): Key for the mel spectrograms in the H5 file
            lazy_load (bool): Whether to use lazy loading
        """
        self.h5_path = h5_path
        self.data_key = data_key
        self.lazy_load = lazy_load
        
        # Get file handle using H5FileManager for lazy loading
        if lazy_load:
            h5_manager = H5FileManager.get_instance()
            h5_file = h5_manager.get_file(h5_path)
        else:
            h5_file = h5py.File(h5_path, 'r')
        
        # Check if data key exists
        if data_key not in h5_file:
            raise KeyError(f"Data key '{data_key}' not found in {h5_path}")
        
        # Get dataset info
        self.data_shape = h5_file[data_key].shape
        self.num_samples = self.data_shape[0]
        
        # Print dataset information
        print(f"Loaded mel spectrogram dataset from {h5_path}")
        print(f"Data shape: {self.data_shape}")
        print(f"Number of samples: {self.num_samples}")
        
        # If not using lazy loading, load all data at once
        if not lazy_load:
            self.data = h5_file[data_key][:]
            h5_file.close()
    
    def __len__(self):
        """
        Get the number of items in the dataset.
        
        Returns:
            int: Number of items
        """
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            torch.Tensor: Mel spectrogram tensor of shape [1, F, T]
        """
        # Get mel spectrogram
        if self.lazy_load:
            # Lazy loading - get from H5 file
            h5_manager = H5FileManager.get_instance()
            h5_file = h5_manager.get_file(self.h5_path)
            mel_spec = h5_file[self.data_key][idx]
        else:
            # Already loaded - get from memory
            mel_spec = self.data[idx]
        
        # Convert to tensor and add channel dimension
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
        
        # Get H5 file path from config
        self.h5_path = os.path.join(
            config['data']['bin_dir'],
            config['data']['bin_file']
        )
        
        # Set data key - default to 'mel_spectrograms'
        self.data_key = config['data'].get('data_key', 'mel_spectrograms')
        
        # Lazy loading setting - default to True for multi-worker dataloaders
        self.lazy_load = config['data'].get('lazy_load', self.num_workers == 0)
        
        # Track dataset instances
        self.dataset = None
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
        Setup train/val datasets.
        """
        if self.dataset is None:
            # Create dataset
            self.dataset = MelSpectrogramDataset(
                h5_path=self.h5_path,
                data_key=self.data_key,
                lazy_load=self.lazy_load
            )
            
            # Split into train and validation sets
            val_size = int(len(self.dataset) * self.validation_split)
            train_size = len(self.dataset) - val_size
            
            # Create a generator with fixed seed for reproducibility
            generator = torch.Generator().manual_seed(42)
            
            # Split dataset
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, 
                [train_size, val_size],
                generator=generator
            )
            
            print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
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
            collate_fn=collate_fn,
            persistent_workers=True
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
            collate_fn=collate_fn,
            persistent_workers=True
        )
        
    def teardown(self, stage=None):
        """
        Clean up after the training/testing is finished.
        
        Args:
            stage (str, optional): 'fit', 'validate', 'test', or 'predict'
        """
        # Close any open H5 files when done
        if stage == 'fit' or stage is None:
            H5FileManager.get_instance().close_all()