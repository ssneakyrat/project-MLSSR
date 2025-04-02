import torch
from data.datasets import BaseDataset
from utils.transform import batch_prepare_mel_spectrograms

class MelSpectrogramDataset(BaseDataset):
    """
    Dataset for Mel Spectrogram reconstruction.
    """
    def __init__(self, data_items, target_length=128, target_bins=80):
        """
        Args:
            data_items (list): List of data items from load_dataset
            target_length (int): Target number of time frames
            target_bins (int): Target number of mel bins
        """
        super().__init__(data_items)
        self.target_length = target_length
        self.target_bins = target_bins
    
    def __getitem__(self, idx):
        # Get mel spectrogram
        item = self.data_items[idx]
        mel_spec = item['mel_spec']
        
        # Process mel spectrogram
        mel_tensor = batch_prepare_mel_spectrograms([mel_spec], self.target_length, self.target_bins)
        
        # Return as a single tensor (removing batch dimension)
        return mel_tensor.squeeze(0)

def collate_fn(batch):
    """
    Collate function for DataLoader.
    """
    # Batch is already properly processed, just stack them
    return torch.stack(batch, dim=0)