import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import h5py
import numpy as np
import math

from data.dataset import H5Dataset, H5FileManager

class ConditionalMelDataset(H5Dataset):
    """Dataset that loads mel spectrograms along with conditioning data"""
    def __init__(self, h5_path, data_key, max_frames=None, transform=None, 
                 lazy_load=True, variable_length=True, return_conditioning=True):
        super().__init__(h5_path, data_key, transform, lazy_load, variable_length)
        self.return_conditioning = return_conditioning
        self.max_frames = max_frames
        
        # Load phone map if available
        h5_manager = H5FileManager.get_instance()
        h5_file = h5_manager.get_file(h5_path)
        
        if 'phone_map' in h5_file:
            self.phone_map = list(h5_file['phone_map'][:])
        else:
            self.phone_map = []
        
        # Check if we have phone information
        self.has_phone_data = ('phone_frame_starts' in h5_file and 
                              'phone_frame_ends' in h5_file and 
                              'phone_texts' in h5_file)
        
        # Check if we have pitch information
        self.has_pitch_data = 'MIDI_PITCH' in h5_file
        
        # Check if we have F0 information
        self.has_f0_data = 'f0_values' in h5_file
        
    def __getitem__(self, idx):
        # Get mel spectrogram
        mel = super().__getitem__(idx)
        
        # Debug the shape of the mel tensor
        if not isinstance(mel, torch.Tensor):
            print(f"Warning: mel is not a tensor but {type(mel)}")
            mel = torch.tensor(mel) if hasattr(mel, '__array__') else torch.zeros((1, 80, 100))
        
        # Ensure mel has the right dimensions
        if mel.dim() == 2:  # If mel is [freq_bins, time_frames]
            mel = mel.unsqueeze(0)  # Convert to [1, freq_bins, time_frames]
        elif mel.dim() == 1:  # If mel is somehow flattened
            print(f"Warning: mel has only 1 dimension with shape {mel.shape}")
            # Attempt to reshape assuming standard dimensions
            mel = mel.reshape(1, -1, 100)  # Reshape with a default time dimension
        
        if not self.return_conditioning:
            return mel
        
        # Initialize conditioning dictionary
        conditioning = {}
        
        # Get h5 file
        h5_manager = H5FileManager.get_instance()
        h5_file = h5_manager.get_file(self.h5_path)
        
        # Get actual mel length (remove padding)
        try:
            if self.variable_length:
                if mel.dim() >= 3:  # Make sure mel has enough dimensions
                    time_frames = min(mel.shape[2], h5_file['lengths'][idx])
                else:
                    time_frames = h5_file['lengths'][idx]
            else:
                if mel.dim() >= 3:  # Make sure mel has enough dimensions
                    time_frames = mel.shape[2]
                else:
                    # Default to a reasonable value if dimensions are wrong
                    time_frames = 100  # Or some default value
        except Exception as e:
            print(f"Error determining time_frames: {e}. Using default value.")
            time_frames = 100  # Default value if we can't determine
        
        # Get phoneme information
        if self.has_phone_data:
            try:
                phone_starts = h5_file['phone_frame_starts'][idx]
                phone_ends = h5_file['phone_frame_ends'][idx]
                phone_texts = h5_file['phone_texts'][idx]
                
                # Convert phoneme text to phoneme IDs
                phone_ids = []
                for phone in phone_texts:
                    if phone in self.phone_map:
                        phone_ids.append(self.phone_map.index(phone))
                    else:
                        phone_ids.append(0)  # Unknown phone
                
                # Truncate to max length if needed
                if self.max_frames and time_frames > self.max_frames:
                    time_frames = self.max_frames
                    # Find last phoneme that fits
                    last_idx = 0
                    for i, end in enumerate(phone_ends):
                        if end <= time_frames:
                            last_idx = i
                        else:
                            break
                    
                    # Truncate phoneme data
                    phone_starts = phone_starts[:last_idx+1]
                    phone_ends = phone_ends[:last_idx+1]
                    phone_ids = phone_ids[:last_idx+1]
                    
                    # Ensure last phoneme doesn't go beyond time_frames
                    if phone_ends[-1] > time_frames:
                        phone_ends[-1] = time_frames
                
                # Calculate phoneme durations
                durations = phone_ends - phone_starts
                
                # Store in conditioning dictionary
                conditioning['phoneme_ids'] = torch.tensor(phone_ids)
                conditioning['phoneme_starts'] = torch.tensor(phone_starts)
                conditioning['phoneme_ends'] = torch.tensor(phone_ends)
                conditioning['phoneme_durations'] = torch.tensor(durations)
                
            except Exception as e:
                print(f"Error loading phoneme data: {e}")
                # Create empty tensors for phoneme data
                conditioning['phoneme_ids'] = torch.zeros(1, dtype=torch.long)
                conditioning['phoneme_starts'] = torch.zeros(1, dtype=torch.long)
                conditioning['phoneme_ends'] = torch.ones(1, dtype=torch.long)
                conditioning['phoneme_durations'] = torch.ones(1, dtype=torch.long)
        
        # Get pitch information
        if self.has_pitch_data:
            try:
                midi_pitches = h5_file['MIDI_PITCH'][idx]
                
                # Convert to frame-level representation
                frame_midi = np.zeros(time_frames, dtype=np.int32)
                
                # Fill in frame-level pitch values
                if len(phone_starts) > 0 and len(midi_pitches) > 0:
                    for i, (start, end, pitch) in enumerate(zip(phone_starts, phone_ends, midi_pitches)):
                        if i >= len(midi_pitches):
                            break
                        # Ensure within bounds
                        start = max(0, min(start, time_frames - 1))
                        end = max(start + 1, min(end, time_frames))
                        
                        # Fill in the range with the pitch value
                        frame_midi[start:end] = pitch
                
                # Store in conditioning dictionary
                conditioning['midi_pitch'] = torch.tensor(frame_midi)
                
            except Exception as e:
                print(f"Error loading MIDI pitch data: {e}")
                conditioning['midi_pitch'] = torch.zeros(time_frames, dtype=torch.long)
        
        # Get F0 information
        if self.has_f0_data:
            try:
                f0_values = h5_file['f0_values'][idx]
                
                # Truncate to actual length
                f0_values = f0_values[:time_frames]
                
                # Store in conditioning dictionary
                conditioning['f0'] = torch.tensor(f0_values)
                
            except Exception as e:
                print(f"Error loading F0 data: {e}")
                conditioning['f0'] = torch.zeros(time_frames, dtype=torch.float)
        
        return mel, conditioning

def collate_conditional_batch(batch):
    """
    Custom collate function for conditional batch
    
    Args:
        batch: List of (mel, conditioning) tuples
        
    Returns:
        Tuple of (mel_batch, conditioning_batch, mask)
    """
    # Error checking
    if not batch:
        print("Warning: Empty batch received in collate_conditional_batch")
        return torch.zeros((0, 1, 80, 100)), {}, torch.zeros((0, 100), dtype=torch.bool)
    
    # Check batch structure
    if isinstance(batch[0], tuple) and len(batch[0]) == 2:
        # We have (mel, conditioning) pairs
        mels = [item[0] for item in batch]
        cond_batch = [item[1] for item in batch]
        
        # Handle dimension errors
        for i in range(len(mels)):
            if not isinstance(mels[i], torch.Tensor):
                print(f"Warning: item {i} is not a tensor but {type(mels[i])}")
                mels[i] = torch.zeros((1, 80, 100))
            
            # Ensure 3D tensors for all melspectrograms
            if mels[i].dim() == 2:
                mels[i] = mels[i].unsqueeze(0)  # Add channel dimension
            elif mels[i].dim() == 1:
                mels[i] = mels[i].reshape(1, 80, -1)  # Reshape to standard format
            elif mels[i].dim() != 3:
                print(f"Warning: item {i} has unexpected dimensions {mels[i].shape}")
                mels[i] = torch.zeros((1, 80, 100))
        
        # Process mels
        batch_size = len(mels)
        
        # Make sure all mels have consistent dimensions
        try:
            channels = mels[0].shape[0]
            height = mels[0].shape[1]  # freq bins
            max_length = max([item.shape[2] for item in mels])
        except (IndexError, AttributeError) as e:
            print(f"Error processing mel dimensions: {e}")
            # Use default values
            channels = 1
            height = 80
            max_length = 100
        
        # Create tensor to hold the batch - [batch, channels, height, max_length]
        mel_batch = torch.zeros((batch_size, channels, height, max_length))
        
        # Create a mask to track actual sequence lengths (1 for data, 0 for padding)
        mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        
        # Fill in the batch tensor
        for i, item in enumerate(mels):
            # Get sequence length (time dimension)
            seq_len = item.shape[2]
            # Place the data in the batch tensor
            mel_batch[i, :, :, :seq_len] = item
            # Mark actual data positions in the mask
            mask[i, :seq_len] = 1
        
        # Process conditioning data
        conditioning_batch = {}
        
        # Initialize conditioning batch with empty tensors
        if cond_batch and len(cond_batch) > 0:
            conditioning_keys = cond_batch[0].keys()
            for key in conditioning_keys:
                if key in ['phoneme_ids', 'phoneme_durations', 'phoneme_starts', 'phoneme_ends']:
                    # For phoneme data, we need to pad to max phoneme length
                    try:
                        max_phones = max([len(item[key]) for item in cond_batch])
                        conditioning_batch[key] = torch.zeros((batch_size, max_phones), dtype=cond_batch[0][key].dtype)
                        
                        for i, item in enumerate(cond_batch):
                            phone_len = len(item[key])
                            conditioning_batch[key][i, :phone_len] = item[key]
                    except Exception as e:
                        print(f"Error processing conditioning key {key}: {e}")
                        # Create a fallback
                        conditioning_batch[key] = torch.zeros((batch_size, 1), dtype=torch.long)
                
                elif key in ['midi_pitch', 'f0']:
                    # For frame-level data, we need to pad to max frame length
                    try:
                        conditioning_batch[key] = torch.zeros((batch_size, max_length), dtype=cond_batch[0][key].dtype)
                        
                        for i, item in enumerate(cond_batch):
                            if key in item:
                                frame_len = min(len(item[key]), max_length)
                                conditioning_batch[key][i, :frame_len] = item[key][:frame_len]
                    except Exception as e:
                        print(f"Error processing conditioning key {key}: {e}")
                        # Create a fallback
                        conditioning_batch[key] = torch.zeros((batch_size, max_length), dtype=torch.float)
        
        return mel_batch, conditioning_batch, mask
    else:
        # We only have mels (no conditioning)
        print("Warning: batch doesn't contain conditioning data. Using empty conditioning.")
        # Process mels only
        if not isinstance(batch[0], torch.Tensor):
            print(f"Warning: batch items are not tensors but {type(batch[0])}")
            # Create a dummy tensor batch
            return torch.zeros((len(batch), 1, 80, 100)), {}, torch.zeros((len(batch), 100), dtype=torch.bool)
        
        # Process mels like the original collate_variable_length
        batch_size = len(batch)
        
        # Ensure all items are 3D tensors
        for i in range(len(batch)):
            if batch[i].dim() == 2:
                batch[i] = batch[i].unsqueeze(0)
            elif batch[i].dim() == 1:
                batch[i] = batch[i].reshape(1, 80, -1)
            elif batch[i].dim() != 3:
                batch[i] = torch.zeros((1, 80, 100))
        
        try:
            channels = batch[0].shape[0]
            height = batch[0].shape[1]  # freq bins
            max_length = max([item.shape[2] for item in batch])
        except (IndexError, AttributeError) as e:
            print(f"Error processing tensor dimensions: {e}")
            channels = 1
            height = 80
            max_length = 100
        
        # Create tensor to hold the batch - [batch, channels, height, max_length]
        mel_batch = torch.zeros((batch_size, channels, height, max_length))
        
        # Create a mask to track actual sequence lengths (1 for data, 0 for padding)
        mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        
        # Fill in the batch tensor
        for i, item in enumerate(batch):
            # Get sequence length (time dimension)
            seq_len = item.shape[2]
            # Place the data in the batch tensor
            mel_batch[i, :, :, :seq_len] = item
            # Mark actual data positions in the mask
            mask[i, :seq_len] = 1
        
        return mel_batch, {}, mask

class ConditionalDataModule(pl.LightningDataModule):
    """Data module for conditional mel spectrograms"""
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
        
        # Handle variable length inputs
        self.variable_length = config['data'].get('variable_length', False)
        
        # Calculate max time frames for audio
        if 'max_audio_length' in config['audio']:
            self.max_audio_length = config['audio']['max_audio_length']
            sample_rate = config['audio']['sample_rate']
            hop_length = config['audio']['hop_length']
            self.max_frames = math.ceil(self.max_audio_length * sample_rate / hop_length)
        else:
            self.max_frames = config['model'].get('time_frames', 128)
        
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        
    def setup(self, stage=None):
        if self.dataset is None:
            self.dataset = ConditionalMelDataset(
                h5_path=self.h5_path,
                data_key=self.data_key,
                max_frames=self.max_frames,
                lazy_load=self.lazy_load,
                variable_length=self.variable_length,
                return_conditioning=True
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
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_conditional_batch
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            collate_fn=collate_conditional_batch
        )
        
    def teardown(self, stage=None):
        if stage == 'fit' or stage is None:
            H5FileManager.get_instance().close_all()