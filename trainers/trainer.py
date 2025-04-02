import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from models.unet import UNet
from models.losses import CombinedLoss
from utils.transforms import batch_prepare_mel_spectrograms
from trainers.validator import validate

class MelSpectrogramDataset(Dataset):
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
        self.data_items = data_items
        self.target_length = target_length
        self.target_bins = target_bins
    
    def __len__(self):
        return len(self.data_items)
    
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

def train(config, data_items, save_dir='runs/unet', num_epochs=100, batch_size=32,
          validation_split=0.1, log_interval=10, save_interval=10, device=None):
    """
    Train the U-Net model for mel spectrogram reconstruction.
    
    Args:
        config (dict): Configuration dictionary
        data_items (list): List of data items from load_dataset
        save_dir (str): Directory to save checkpoints and logs
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size
        validation_split (float): Fraction of data to use for validation
        log_interval (int): Interval for logging training statistics
        save_interval (int): Interval for saving model checkpoints
        device (torch.device): Device to use for training (default: auto-detect)
    """
    # Create directories
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create TensorBoard writer
    writer = SummaryWriter(save_dir)
    
    # Split data into training and validation sets
    num_val = int(len(data_items) * validation_split)
    indices = list(range(len(data_items)))
    np.random.shuffle(indices)
    train_indices = indices[num_val:]
    val_indices = indices[:num_val]
    
    train_data = [data_items[i] for i in train_indices]
    val_data = [data_items[i] for i in val_indices]
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create datasets and dataloaders
    train_dataset = MelSpectrogramDataset(train_data)
    val_dataset = MelSpectrogramDataset(val_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model, loss function, and optimizer
    model = UNet().to(device)
    criterion = CombinedLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Log model architecture to TensorBoard
    dummy_input = torch.zeros(1, 1, 128, 80).to(device)
    writer.add_graph(model, dummy_input)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    print("Beginning training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            global_step += 1
            epoch_loss += loss.item()
            
            # Log to TensorBoard
            if batch_idx % log_interval == 0:
                writer.add_scalar('Loss/train/batch', loss.item(), global_step)
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.6f}")
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train/epoch', avg_epoch_loss, epoch)
        
        # Validation
        val_loss, val_images = validate(model, val_loader, criterion, device)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        
        # Log sample images
        for i, (input_img, output_img, comparison) in enumerate(val_images[:5]):  # Log up to 5 samples
            writer.add_image(f'Sample {i+1}/Input', input_img, epoch)
            writer.add_image(f'Sample {i+1}/Output', output_img, epoch)
            writer.add_image(f'Sample {i+1}/Comparison', comparison, epoch)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save model checkpoint
        if epoch % save_interval == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(checkpoint_dir, f'unet_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'val_loss': val_loss,
                'global_step': global_step
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, 'unet_best.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'val_loss': val_loss,
                'global_step': global_step
            }, best_model_path)
            print(f"Best model saved with validation loss: {val_loss:.6f}")
        
        # Print epoch summary
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {elapsed_time:.2f}s, "
              f"Train Loss: {avg_epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Close TensorBoard writer
    writer.close()
    print("Training complete!")
    
    return model, best_val_loss