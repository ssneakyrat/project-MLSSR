import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import torchvision.transforms as T

def create_comparison_image(input_mel, output_mel):
    """
    Create a comparison image showing input and output mel spectrograms side by side.
    
    Args:
        input_mel (torch.Tensor): Input mel spectrogram [1, T, F]
        output_mel (torch.Tensor): Output (reconstructed) mel spectrogram [1, T, F]
        
    Returns:
        tuple: (input_tensor, output_tensor, comparison_tensor) normalized image tensors
               in the format expected by TensorBoard (C, H, W)
    """
    # Convert to numpy arrays
    input_np = input_mel.squeeze(0).cpu().numpy()
    output_np = output_mel.squeeze(0).cpu().numpy()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot input spectrogram
    im1 = ax1.imshow(input_np, aspect='auto', origin='lower', cmap='viridis')
    ax1.set_title('Input Mel-Spectrogram')
    ax1.set_ylabel('Mel Bins')
    ax1.set_xlabel('Time Frames')
    
    # Plot output spectrogram
    im2 = ax2.imshow(output_np, aspect='auto', origin='lower', cmap='viridis')
    ax2.set_title('Reconstructed Mel-Spectrogram')
    ax2.set_ylabel('Mel Bins')
    ax2.set_xlabel('Time Frames')
    
    # Add colorbar
    plt.colorbar(im1, ax=ax1, format='%+2.0f dB')
    plt.colorbar(im2, ax=ax2, format='%+2.0f dB')
    
    # Adjust layout
    plt.tight_layout()
    
    # Convert figure to image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    # Convert to PIL Image and then to tensor
    from PIL import Image
    comparison_img = Image.open(buf)
    to_tensor = T.ToTensor()
    comparison_tensor = to_tensor(comparison_img)
    
    # Create separate input and output tensors for individual TensorBoard logging
    # Normalize each tensor to [0, 1] for TensorBoard display
    input_tensor = torch.from_numpy(input_np).unsqueeze(0)  # Add channel dimension
    input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min() + 1e-8)
    
    output_tensor = torch.from_numpy(output_np).unsqueeze(0)  # Add channel dimension
    output_tensor = (output_tensor - output_tensor.min()) / (output_tensor.max() - output_tensor.min() + 1e-8)
    
    return input_tensor, output_tensor, comparison_tensor

def validate(model, val_loader, criterion, device):
    """
    Validate the model and generate validation visualizations.
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use for validation
        
    Returns:
        tuple: (validation_loss, validation_images) where validation_images is a list of
               tuples (input_tensor, output_tensor, comparison_tensor)
    """
    model.eval()
    val_loss = 0.0
    val_images = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch = batch.to(device)
            
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch)
            
            # Update statistics
            val_loss += loss.item()
            
            # Generate validation images for the first few samples in the batch
            if batch_idx == 0:
                for i in range(min(5, batch.size(0))):  # Up to 5 samples
                    input_mel = batch[i].unsqueeze(0)  # Add batch dimension back
                    output_mel = outputs[i].unsqueeze(0)
                    
                    # Create comparison image
                    input_tensor, output_tensor, comparison_tensor = create_comparison_image(
                        input_mel, output_mel
                    )
                    
                    val_images.append((input_tensor, output_tensor, comparison_tensor))
    
    # Calculate average validation loss
    avg_val_loss = val_loss / len(val_loader)
    
    return avg_val_loss, val_images