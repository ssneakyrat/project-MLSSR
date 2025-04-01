import argparse
import random
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import io
import torchvision

from data.datasets import load_dataset
from utils.utils_general import load_config, plot_alignment

def validate_dataset(config, log_dir='logs/validation', num_samples=4):
    """
    Validate the dataset module by loading data and creating visualizations.
    
    Args:
        config (dict): Configuration dictionary
        log_dir (str): Directory to save TensorBoard logs
        num_samples (int): Number of random samples to visualize
    """
    print("Validating dataset module...")
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Load dataset
    data_items = load_dataset(split='train', shuffle=True)
    
    # Select random samples (or fewer if less than num_samples items available)
    samples_to_visualize = min(num_samples, len(data_items))
    if samples_to_visualize == 0:
        print("Error: No data items found in the dataset")
        return
    
    random_indices = random.sample(range(len(data_items)), samples_to_visualize)
    
    print(f"Selected {samples_to_visualize} random samples for validation")
    
    # Collect all generated images to display in the same row
    all_images = []
    image_labels = []
    
    # Create visualizations for each sample
    for i, idx in enumerate(random_indices):
        item = data_items[idx]
        print(f"Processing sample {i+1}/{samples_to_visualize}: {item['id']}")
        
        # Check if all required data is available
        if 'mel_spec' not in item or 'f0' not in item:
            print(f"  Missing mel_spec or f0 for sample {item['id']}, skipping...")
            continue
        
        # Create phoneme tuples for visualization
        phoneme_tuples = []
        if 'phone_texts' in item and 'phone_starts' in item and 'phone_ends' in item:
            for j in range(len(item['phone_texts'])):
                phoneme_tuples.append((
                    item['phone_starts'][j],
                    item['phone_ends'][j],
                    item['phone_texts'][j]
                ))
        
        # Create phoneme MIDI tuples if available
        phoneme_midi_tuples = None
        if 'phone_texts' in item and 'phone_starts' in item and 'phone_ends' in item and 'phoneme_midi' in item:
            phoneme_midi_tuples = []
            for j in range(len(item['phone_texts'])):
                phoneme_midi_tuples.append((
                    item['phone_starts'][j],
                    item['phone_ends'][j],
                    item['phone_texts'][j],
                    item['phoneme_midi'][j]
                ))
        
        # Generate the alignment plot directly as bytes
        try:
            # Get plot as bytes
            plot_bytes = plot_alignment(
                item['mel_spec'],
                item['f0'],
                phoneme_tuples,
                config=config,
                phoneme_midi=phoneme_midi_tuples,
                midi_notes=item.get('midi_notes'),
                return_bytes=True
            )
            
            # Convert bytes to image
            img = Image.open(io.BytesIO(plot_bytes))
            img_array = np.array(img)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(img_array.transpose((2, 0, 1)))
            
            # Add to our collection
            all_images.append(img_tensor)
            image_labels.append(f"{item['id']}")
            
            print(f"  Generated visualization for sample {item['id']}")
        except Exception as e:
            print(f"  Error generating visualization: {e}")
    
    # Add images to TensorBoard individually but with organized tags
    if all_images:
        try:
            # Add a summary text
            writer.add_text('Validation Dataset Images', 
                          f"Dataset samples visualization ({len(all_images)} samples)", 
                          global_step=0)
            
            # Resize images to a reasonable resolution
            TARGET_HEIGHT = 600
            
            # Add each image with a consistent tag structure
            for i, (img_tensor, label) in enumerate(zip(all_images, image_labels)):
                # Convert tensor to PIL image for resizing
                img = Image.fromarray(img_tensor.permute(1, 2, 0).numpy().astype(np.uint8))
                
                # Maintain aspect ratio while resizing
                width, height = img.size
                new_width = int((TARGET_HEIGHT / height) * width)
                img_resized = img.resize((new_width, TARGET_HEIGHT), Image.LANCZOS)
                
                # Convert back to tensor
                img_tensor_resized = torch.from_numpy(
                    np.array(img_resized).transpose((2, 0, 1))
                )
                
                # Add individual image with numbered prefix for ordering
                writer.add_image(
                    f'Validation Dataset Images/{i+1:02d}_{label}',
                    img_tensor_resized, 
                    global_step=0,  # Use 0 to avoid slider
                    dataformats='CHW'
                )
            
            # Add sample IDs as text
            id_text = "Sample IDs:\n" + "\n".join([f"• {label}" for label in image_labels])
            writer.add_text('Dataset Sample IDs', id_text, global_step=0)
            
            print(f"Added {len(all_images)} individual visualizations to TensorBoard")
            print("- Click 'Validation Dataset Images' to see and interact with individual images")
        except Exception as e:
            print(f"Error adding visualizations to TensorBoard: {e}")
    
    # Add summary statistics
    writer.add_text("Dataset Summary", f"Total samples in dataset: {len(data_items)}")
    writer.add_text("Validation Info", f"Visualized {samples_to_visualize} random samples")
    
    writer.close()
    print(f"Validation complete. Results saved to {log_dir}")
    print(f"To view visualizations, run: tensorboard --logdir={log_dir}")

def main():
    """
    Main function to parse arguments and run validation.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Validate different modules of the system')
    parser.add_argument('--module', type=str, required=True, 
                        help='Module to validate (e.g., dataset)')
    parser.add_argument('--log_dir', type=str, default='logs/validation',
                        help='Directory to save TensorBoard logs')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of random samples to visualize')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    # Validate the specified module
    if args.module.lower() == 'dataset':
        validate_dataset(config, log_dir=args.log_dir, num_samples=args.num_samples)
    else:
        print(f"Error: Unsupported module '{args.module}'")
        print("Supported modules: dataset")
        print("Example usage: python validate_module.py --module dataset")

if __name__ == "__main__":
    main()