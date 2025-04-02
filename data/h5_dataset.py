import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import h5py
import numpy as np

from utils.utils_datasets import H5FileManager

class H5Dataset(Dataset):
    """
    A generic dataset for loading data from H5 files with lazy loading support.
    """
    def __init__(self, h5_path, data_key, label_key=None, transform=None, lazy_load=True):
        """
        Initialize the dataset.
        
        Args:
            h5_path (str): Path to the H5 file
            data_key (str): Key for the data in the H5 file
            label_key (str, optional): Key for the labels in the H5 file. If None, returns only data.
            transform (callable, optional): Optional transform to apply to the data
            lazy_load (bool): Whether to use lazy loading or load everything in memory
        """
        self.h5_path = h5_path
        self.data_key = data_key
        self.label_key = label_key
        self.transform = transform
        self.lazy_load = lazy_load
        
        # Get the length of the dataset
        if lazy_load:
            h5_manager = H5FileManager.get_instance()
            h5_file = h5_manager.get_file(h5_path)
        else:
            h5_file = h5py.File(h5_path, 'r')
            
        if data_key in h5_file:
            self.length = len(h5_file[data_key])
            # Store shapes for validation
            self.data_shape = h5_file[data_key].shape[1:]
            if label_key and label_key in h5_file:
                self.label_shape = h5_file[label_key].shape[1:]
            else:
                self.label_shape = None
        else:
            raise KeyError(f"Data key '{data_key}' not found in {h5_path}")
            
        # If not using lazy loading, load everything now
        if not lazy_load:
            self.data = h5_file[data_key][:]
            if label_key and label_key in h5_file:
                self.labels = h5_file[label_key][:]
            else:
                self.labels = None
            h5_file.close()
            
        print(f"Initialized H5Dataset with {self.length} samples from {h5_path}")
        print(f"Data shape: {self.data_shape}")
        if self.label_shape:
            print(f"Label shape: {self.label_shape}")
    
    def __len__(self):
        """
        Get the number of items in the dataset.
        
        Returns:
            int: Number of items
        """
        return self.length
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (data, label) if label_key is provided, otherwise just data
        """
        if self.lazy_load:
            # Lazy loading - get data from H5 file
            h5_manager = H5FileManager.get_instance()
            h5_file = h5_manager.get_file(self.h5_path)
            
            data = h5_file[self.data_key][idx]
            # Convert to tensor
            data = torch.from_numpy(data).float()
            
            # Get label if available
            if self.label_key and self.label_key in h5_file:
                label = h5_file[self.label_key][idx]
                label = torch.from_numpy(label).float()
            else:
                label = None
        else:
            # Data already loaded in __init__
            data = torch.from_numpy(self.data[idx]).float()
            if self.labels is not None:
                label = torch.from_numpy(self.labels[idx]).float()
            else:
                label = None
                
        # Apply transform if provided
        if self.transform:
            data = self.transform(data)
            
        # Return data and label if available
        if label is not None:
            return data, label
        else:
            return data


class H5DataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for H5 datasets.
    """
    def __init__(self, h5_path, data_key, label_key=None, batch_size=32, 
                 validation_split=0.2, test_split=0.0, transform=None,
                 num_workers=4, pin_memory=True, lazy_load=True,
                 max_samples=None, sample_percentage=None):
        """
        Initialize the data module.
        
        Args:
            h5_path (str): Path to the H5 file
            data_key (str): Key for the data in the H5 file
            label_key (str, optional): Key for the labels in the H5 file
            batch_size (int): Batch size for data loading
            validation_split (float): Fraction of data to use for validation
            test_split (float): Fraction of data to use for testing
            transform (callable, optional): Optional transform to apply to the data
            num_workers (int): Number of workers for data loading
            pin_memory (bool): Whether to pin memory for faster GPU transfer
            lazy_load (bool): Whether to use lazy loading or load everything in memory
            max_samples (int, optional): Maximum number of samples to use from dataset
            sample_percentage (float, optional): Percentage of dataset to use (0.0-1.0)
        """
        super().__init__()
        self.h5_path = h5_path
        self.data_key = data_key
        self.label_key = label_key
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.transform = transform
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.lazy_load = lazy_load
        self.max_samples = max_samples
        self.sample_percentage = sample_percentage
        
        # Initialize datasets to None
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
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
        """
        if self.dataset is None:
            # Create the dataset
            self.dataset = H5Dataset(
                self.h5_path,
                self.data_key,
                self.label_key,
                self.transform,
                self.lazy_load
            )
            
            # Apply dataset subsetting if requested
            full_dataset_size = len(self.dataset)
            subset_size = full_dataset_size
            
            # Priority: max_samples takes precedence over sample_percentage
            if self.max_samples is not None and self.max_samples > 0:
                subset_size = min(self.max_samples, full_dataset_size)
            elif self.sample_percentage is not None and 0.0 < self.sample_percentage <= 1.0:
                subset_size = int(full_dataset_size * self.sample_percentage)
            
            # Create a subset if needed
            if subset_size < full_dataset_size:
                print(f"Creating dataset subset: {subset_size}/{full_dataset_size} samples ({subset_size/full_dataset_size:.1%})")
                
                # Create a generator with fixed seed for reproducibility
                generator = torch.Generator().manual_seed(42)
                
                # Get indices for the subset
                indices = torch.randperm(full_dataset_size, generator=generator)[:subset_size].tolist()
                
                # Create a Subset dataset
                self.dataset = torch.utils.data.Subset(self.dataset, indices)
            
            # Calculate split sizes
            dataset_size = len(self.dataset)
            val_size = int(dataset_size * self.validation_split)
            test_size = int(dataset_size * self.test_split)
            train_size = dataset_size - val_size - test_size
            
            # Create a generator with fixed seed for reproducibility
            generator = torch.Generator().manual_seed(42)
            
            # Split the dataset
            if test_size > 0:
                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    self.dataset, 
                    [train_size, val_size, test_size],
                    generator=generator
                )
                print(f"Dataset split: {train_size} train, {val_size} validation, {test_size} test")
            else:
                self.train_dataset, self.val_dataset = random_split(
                    self.dataset, 
                    [train_size, val_size],
                    generator=generator
                )
                print(f"Dataset split: {train_size} train, {val_size} validation")
    
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
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        
    def test_dataloader(self):
        """
        Create the test dataloader.
        
        Returns:
            DataLoader: Test dataloader
        """
        if self.test_dataset is not None:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
        return None
    
    def teardown(self, stage=None):
        """
        Clean up after the training/testing is finished.
        
        Args:
            stage (str, optional): 'fit', 'validate', 'test', or 'predict'
        """
        # Close any open H5 files when done
        if stage == 'fit' or stage is None:
            H5FileManager.get_instance().close_all()