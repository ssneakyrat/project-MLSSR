import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import h5py
import numpy as np
import math

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
    def __init__(self, h5_path, data_key, transform=None, lazy_load=True, variable_length=False):
        self.h5_path = h5_path
        self.data_key = data_key
        self.transform = transform
        self.lazy_load = lazy_load
        self.variable_length = variable_length
        
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
        
        # Store audio config attributes if available
        if lazy_load:
            if data_key in h5_file and hasattr(h5_file[data_key], 'attrs'):
                self.attrs = dict(h5_file[data_key].attrs)
            else:
                self.attrs = {}
        else:
            self.attrs = {}
    
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
        
        # Make sure data is in the correct format [freq_bins, time_frames]
        if data.dim() == 2:
            # Check if the dimensions are flipped (time_frames, freq_bins)
            if data.shape[0] > data.shape[1]:  # If time > freq, transpose
                data = data.transpose(0, 1)
                
            # Add channel dimension [1, freq_bins, time_frames]
            data = data.unsqueeze(0)
        
        return data

class VariableLengthMelDataset(H5Dataset):
    """Dataset that supports variable length mel spectrograms with max length constraint"""
    def __init__(self, h5_path, data_key, max_frames=None, transform=None, lazy_load=True):
        super().__init__(h5_path, data_key, transform, lazy_load, variable_length=True)
        self.max_frames = max_frames
        
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        
        # If data is already 3D, make sure it's in the correct format [channels, freq, time]
        if data.dim() == 3:
            # Check if the dimensions are flipped [channels, time, freq]
            if data.shape[1] > data.shape[2]:  # If time > freq, transpose
                data = data.transpose(1, 2)
        elif data.dim() == 2:
            # For 2D tensors [freq, time] or [time, freq]
            if data.shape[0] > data.shape[1]:  # If time > freq, transpose
                data = data.transpose(0, 1)
            # Add channel dimension
            data = data.unsqueeze(0)
            
        # Check if we need to limit the time frames
        if self.max_frames is not None and data.shape[2] > self.max_frames:
            # Trim to max_frames (in dim 2, which is time dimension)
            data = data[:, :, :self.max_frames]
        
        return data

class MelAudioDataset(Dataset):
    """Dataset that loads both mel spectrograms and audio waveforms"""
    def __init__(self, h5_path, mel_key='mel_spectrograms', audio_key='waveforms', 
                 max_mel_frames=None, max_audio_samples=None, lazy_load=True):
        self.h5_path = h5_path
        self.mel_key = mel_key
        self.audio_key = audio_key
        self.max_mel_frames = max_mel_frames
        self.max_audio_samples = max_audio_samples
        self.lazy_load = lazy_load
        
        if lazy_load:
            h5_manager = H5FileManager.get_instance()
            self.h5_file = h5_manager.get_file(h5_path)
        else:
            self.h5_file = h5py.File(h5_path, 'r')
        
        # Verify keys exist
        if mel_key not in self.h5_file:
            raise KeyError(f"Mel spectrogram key '{mel_key}' not found in {h5_path}")
        if audio_key not in self.h5_file:
            raise KeyError(f"Audio key '{audio_key}' not found in {h5_path}")
        
        self.length = len(self.h5_file[mel_key])
        
        # Store config attributes
        self.mel_attrs = dict(self.h5_file[mel_key].attrs) if hasattr(self.h5_file[mel_key], 'attrs') else {}
        self.audio_attrs = dict(self.h5_file[audio_key].attrs) if hasattr(self.h5_file[audio_key], 'attrs') else {}
        
        # Check if we have length information
        self.lengths_available = 'lengths' in self.h5_file
        
        # If not lazy loading, load all data into memory
        if not lazy_load:
            self.mel_data = self.h5_file[mel_key][:]
            self.audio_data = self.h5_file[audio_key][:]
            if self.lengths_available:
                self.lengths = self.h5_file['lengths'][:]
            self.h5_file.close()
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load mel spectrogram
        if self.lazy_load:
            mel_data = self.h5_file[self.mel_key][idx]
            audio_data = self.h5_file[self.audio_key][idx]
            
            # Get actual lengths if available
            if self.lengths_available:
                mel_length, audio_length = self.h5_file['lengths'][idx]
            else:
                mel_length, audio_length = mel_data.shape[1], len(audio_data)
        else:
            mel_data = self.mel_data[idx]
            audio_data = self.audio_data[idx]
            
            # Get actual lengths if available
            if self.lengths_available:
                mel_length, audio_length = self.lengths[idx]
            else:
                mel_length, audio_length = mel_data.shape[1], len(audio_data)
        
        # Convert to torch tensors
        mel_tensor = torch.from_numpy(mel_data).float()
        
        # Apply max length constraints if specified
        if self.max_mel_frames is not None and mel_tensor.shape[1] > self.max_mel_frames:
            mel_tensor = mel_tensor[:, :self.max_mel_frames]
            mel_length = min(mel_length, self.max_mel_frames)
        
        # Add channel dimension for mel: [freq, time] -> [1, freq, time]
        mel_tensor = mel_tensor.unsqueeze(0)
        
        # Process audio data
        audio_tensor = torch.from_numpy(audio_data[:audio_length]).float()
        
        if self.max_audio_samples is not None and audio_tensor.size(0) > self.max_audio_samples:
            audio_tensor = audio_tensor[:self.max_audio_samples]
        
        # Normalize audio to [-1, 1] if needed (assuming int16 input)
        if audio_tensor.dtype == torch.int16 or (audio_tensor.max() > 1.0 or audio_tensor.min() < -1.0):
            audio_tensor = audio_tensor / 32767.0
        
        # Add channel dimension for audio: [samples] -> [1, samples]
        audio_tensor = audio_tensor.unsqueeze(0)
        
        return mel_tensor, audio_tensor

def collate_variable_length(batch):
    """Custom collate function for variable length mel spectrograms"""
    # Check if batch contains tuples (mel, audio)
    has_audio = isinstance(batch[0], tuple) and len(batch[0]) == 2
    
    if has_audio:
        # Separate mel and audio
        mel_batch = [item[0] for item in batch]
        audio_batch = [item[1] for item in batch]
        
        # Find max lengths
        max_mel_length = max([item.shape[2] for item in mel_batch])
        max_audio_length = max([item.shape[1] for item in audio_batch])
        
        batch_size = len(batch)
        mel_channels = mel_batch[0].shape[0]
        mel_height = mel_batch[0].shape[1]  # freq bins
        audio_channels = audio_batch[0].shape[0]  # typically 1
        
        # Create tensors to hold the batch
        batched_mel = torch.zeros((batch_size, mel_channels, mel_height, max_mel_length))
        batched_audio = torch.zeros((batch_size, audio_channels, max_audio_length))
        
        # Create masks to track actual sequence lengths
        mel_mask = torch.zeros((batch_size, max_mel_length), dtype=torch.bool)
        audio_mask = torch.zeros((batch_size, max_audio_length), dtype=torch.bool)
        
        # Fill in the batch tensors
        for i, (mel, audio) in enumerate(zip(mel_batch, audio_batch)):
            # Get sequence lengths
            mel_seq_len = mel.shape[2]
            audio_seq_len = audio.shape[1]
            
            # Place the data in the batch tensors
            batched_mel[i, :, :, :mel_seq_len] = mel
            batched_audio[i, :, :audio_seq_len] = audio
            
            # Mark actual data positions in the masks
            mel_mask[i, :mel_seq_len] = 1
            audio_mask[i, :audio_seq_len] = 1
        
        return (batched_mel, batched_audio), (mel_mask, audio_mask)
    else:
        # Original single-item (mel only) handling
        # Find max length in the batch (time dimension = dim 2)
        max_length = max([item.shape[2] for item in batch])
        batch_size = len(batch)
        channels = batch[0].shape[0]
        height = batch[0].shape[1]  # freq bins
        
        # Create tensor to hold the batch - [batch, channels, height, max_length]
        batched_data = torch.zeros((batch_size, channels, height, max_length))
        
        # Create a mask to track actual sequence lengths (1 for data, 0 for padding)
        mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        
        # Fill in the batch tensor
        for i, item in enumerate(batch):
            # Get sequence length (time dimension)
            seq_len = item.shape[2]
            # Place the data in the batch tensor
            batched_data[i, :, :, :seq_len] = item
            # Mark actual data positions in the mask
            mask[i, :seq_len] = 1
        
        return batched_data, mask

class MelAudioDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['train']['batch_size']
        self.num_workers = config['train'].get('num_workers', 4)
        self.pin_memory = config['train'].get('pin_memory', True)
        self.validation_split = config['train'].get('validation_split', 0.1)
        
        self.h5_path = f"{config['data']['bin_dir']}/{config['data']['bin_file']}"
        self.mel_key = config['data'].get('data_key', 'mel_spectrograms')
        self.audio_key = config['data'].get('audio_key', 'waveforms')
        self.lazy_load = config['data'].get('lazy_load', True)
        self.max_samples = config['data'].get('max_samples', None)
        self.sample_percentage = config['data'].get('sample_percentage', None)
        
        # Handle variable length inputs
        self.variable_length = config['data'].get('variable_length', False)
        
        # Calculate max time frames and audio samples for 10 seconds of audio
        if 'max_audio_length' in config['audio']:
            self.max_audio_length = config['audio']['max_audio_length']
            sample_rate = config['audio']['sample_rate']
            hop_length = config['audio']['hop_length']
            self.max_frames = math.ceil(self.max_audio_length * sample_rate / hop_length)
            self.max_audio_samples = math.ceil(self.max_audio_length * sample_rate)
        else:
            self.max_frames = config['model'].get('time_frames', 128)
            self.max_audio_length = None
            self.max_audio_samples = None
        
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage=None):
        if self.dataset is None:
            # Create the dataset
            try:
                self.dataset = MelAudioDataset(
                    h5_path=self.h5_path,
                    mel_key=self.mel_key,
                    audio_key=self.audio_key,
                    max_mel_frames=self.max_frames,
                    max_audio_samples=self.max_audio_samples,
                    lazy_load=self.lazy_load
                )
                print(f"Created dataset with both mel and audio data")
            except Exception as e:
                print(f"Error creating MelAudioDataset: {e}, falling back to mel-only dataset")
                
                # Fall back to mel-only dataset
                if self.variable_length:
                    self.dataset = VariableLengthMelDataset(
                        h5_path=self.h5_path,
                        data_key=self.mel_key,
                        max_frames=self.max_frames,
                        lazy_load=self.lazy_load
                    )
                else:
                    self.dataset = MelSpectrogramDataset(
                        h5_path=self.h5_path,
                        data_key=self.mel_key,
                        lazy_load=self.lazy_load
                    )
                print(f"Created fallback mel-only dataset")
            
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
            persistent_workers=True,
            collate_fn=collate_variable_length if self.variable_length else None
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True,
            collate_fn=collate_variable_length if self.variable_length else None
        )
        
    def teardown(self, stage=None):
        if stage == 'fit' or stage is None:
            H5FileManager.get_instance().close_all()