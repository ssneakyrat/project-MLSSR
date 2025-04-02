import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import h5py
import numpy as np

class H5FileManager:
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

class H5Dataset(Dataset):
    def __init__(self, h5_path, data_key, transform=None, lazy_load=True):
        self.h5_path = h5_path
        self.data_key = data_key
        self.transform = transform
        self.lazy_load = lazy_load
        
        if lazy_load:
            h5_manager = H5FileManager.get_instance()
            h5_file = h5_manager.get_file(h5_path)
        else:
            h5_file = h5py.File(h5_path, 'r')
            
        if data_key in h5_file:
            self.length = len(h5_file[data_key])
            self.data_shape = h5_file[data_key].shape[1:]
        else:
            raise KeyError(f"Data key '{data_key}' not found in {h5_path}")
            
        if not lazy_load:
            self.data = h5_file[data_key][:]
            h5_file.close()
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.lazy_load:
            h5_manager = H5FileManager.get_instance()
            h5_file = h5_manager.get_file(self.h5_path)
            data = h5_file[self.data_key][idx]
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(self.data[idx]).float()
                
        if self.transform:
            data = self.transform(data)
            
        return data

class MelSpectrogramDataset(H5Dataset):
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return data.unsqueeze(0) if data.dim() == 2 else data

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['train']['batch_size']
        self.num_workers = config['train'].get('num_workers', 4)
        self.pin_memory = config['train'].get('pin_memory', True)
        self.validation_split = config['train'].get('validation_split', 0.1)
        
        self.h5_path = f"{config['data']['bin_dir']}/{config['data']['bin_file']}"
        self.data_key = config['data'].get('data_key', 'mel_spectrograms')
        self.lazy_load = config['data'].get('lazy_load', True)
        self.max_samples = config['data'].get('max_samples', None)
        self.sample_percentage = config['data'].get('sample_percentage', None)
        
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage=None):
        if self.dataset is None:
            self.dataset = MelSpectrogramDataset(
                h5_path=self.h5_path,
                data_key=self.data_key,
                lazy_load=self.lazy_load
            )
            
            full_dataset_size = len(self.dataset)
            subset_size = full_dataset_size
            
            if self.max_samples and self.max_samples > 0:
                subset_size = min(self.max_samples, full_dataset_size)
            elif self.sample_percentage and 0.0 < self.sample_percentage <= 1.0:
                subset_size = int(full_dataset_size * self.sample_percentage)
            
            if subset_size < full_dataset_size:
                generator = torch.Generator().manual_seed(42)
                indices = torch.randperm(full_dataset_size, generator=generator)[:subset_size].tolist()
                self.dataset = torch.utils.data.Subset(self.dataset, indices)
            
            dataset_size = len(self.dataset)
            val_size = int(dataset_size * self.validation_split)
            train_size = dataset_size - val_size
            
            generator = torch.Generator().manual_seed(42)
            
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, 
                [train_size, val_size],
                generator=generator
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True
        )
        
    def teardown(self, stage=None):
        if stage == 'fit' or stage is None:
            H5FileManager.get_instance().close_all()