import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm

from utils.utils_general import load_config
from utils.utils_datasets import load_dataset, H5FileManager

def convert_to_generic_h5(input_h5_path, output_h5_path, data_key='mel_spectrograms', target_shape=None):
    """
    Convert the project-specific H5 format to a more generic format suitable for H5Dataset.
    
    Args:
        input_h5_path (str): Path to the input H5 file
        output_h5_path (str): Path to save the output H5 file
        data_key (str): Key to use for the data in the output H5 file
        target_shape (tuple, optional): Target shape for the data, e.g. (80, 128)
                                       If None, original shapes are preserved.
    """
    print(f"Converting {input_h5_path} to generic H5 format...")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
    
    # Load data items using the existing framework
    data_items = load_dataset(split='train', shuffle=False, lazy_load=True)
    
    if not data_items:
        print("Error: No data items found in the dataset")
        return
    
    print(f"Found {len(data_items)} items in the dataset")
    
    # Create a new H5 file for the generic format
    with h5py.File(output_h5_path, 'w') as out_file:
        # Determine the shape of the first valid item
        if target_shape is None:
            for item in data_items:
                if 'mel_spec' in item and item['mel_spec'] is not None:
                    if isinstance(item['mel_spec'], h5py.Dataset):
                        shape = item['mel_spec'].shape
                    else:
                        shape = item['mel_spec'].shape
                    break
            else:
                print("Error: Could not determine shape from any item")
                return
        else:
            shape = target_shape
        
        print(f"Using shape {shape} for data")
        
        # Create a resizable dataset that we'll extend as we go
        dataset = out_file.create_dataset(
            data_key,
            shape=(0,) + shape,  # Start with 0 items
            maxshape=(None,) + shape,  # Allow unlimited items
            dtype=np.float32,
            chunks=(1,) + shape  # Chunk by individual items
        )
        
        # Optionally create datasets for additional information
        # For example, we could add metadata about each item
        item_ids = out_file.create_dataset(
            'item_ids',
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.special_dtype(vlen=str)
        )
        
        # Process each item
        valid_count = 0
        skipped_count = 0
        
        with tqdm(total=len(data_items), desc="Processing items", unit="item") as pbar:
            for idx, item in enumerate(data_items):
                try:
                    # Check if the item has mel spectrogram data
                    if 'mel_spec' in item and item['mel_spec'] is not None:
                        # Load the mel spectrogram
                        if isinstance(item['mel_spec'], h5py.Dataset):
                            mel_spec = item['mel_spec'][:]
                        else:
                            mel_spec = item['mel_spec']
                        
                        # Reshape if necessary
                        if target_shape is not None and mel_spec.shape != target_shape:
                            from utils.utils_transform import pad_or_truncate_mel
                            mel_spec = pad_or_truncate_mel(mel_spec, target_shape[1], target_shape[0])
                        
                        # Normalize if needed
                        if np.max(mel_spec) > 1.0 or np.min(mel_spec) < 0.0:
                            from utils.utils_transform import normalize_mel_spectrogram
                            mel_spec = normalize_mel_spectrogram(mel_spec)
                        
                        # Resize the dataset to add one more item
                        dataset.resize(valid_count + 1, axis=0)
                        item_ids.resize(valid_count + 1, axis=0)
                        
                        # Add the data
                        dataset[valid_count] = mel_spec
                        item_ids[valid_count] = item.get('id', f"item_{idx}")
                        
                        valid_count += 1
                    else:
                        skipped_count += 1
                except Exception as e:
                    print(f"Error processing item {idx}: {e}")
                    skipped_count += 1
                
                pbar.update(1)
        
        # Add some attributes to the dataset for documentation
        dataset.attrs['description'] = "Mel spectrograms extracted from audio files"
        dataset.attrs['shape_info'] = f"Shape: {shape} (frequency bins, time frames)"
        dataset.attrs['normalization'] = "Values are normalized to [0, 1] range"
        
        # Print summary
        print(f"Conversion complete:")
        print(f"  - Total items processed: {len(data_items)}")
        print(f"  - Valid items saved: {valid_count}")
        print(f"  - Items skipped: {skipped_count}")
        print(f"  - Output file saved to: {output_h5_path}")
        print(f"  - Dataset shape: {dataset.shape}")

def main():
    """
    Main function to parse arguments and run conversion.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Convert data to generic H5 format')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save the output H5 file')
    parser.add_argument('--data_key', type=str, default='mel_spectrograms',
                        help='Key to use for the data in the output H5 file')
    parser.add_argument('--target_length', type=int, default=None,
                        help='Target number of time frames (default: preserve original)')
    parser.add_argument('--target_bins', type=int, default=None,
                        help='Target number of mel bins (default: preserve original)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get the path to the binary directory and file
    bin_dir = config['data']['bin_dir']
    bin_file = config['data']['bin_file']
    input_h5_path = os.path.join(bin_dir, bin_file)
    
    # Set target shape if specified
    target_shape = None
    if args.target_length is not None and args.target_bins is not None:
        target_shape = (args.target_bins, args.target_length)
    
    # Run the conversion
    convert_to_generic_h5(input_h5_path, args.output, args.data_key, target_shape)
    
    # Clean up H5 file handles
    H5FileManager.get_instance().close_all()

if __name__ == "__main__":
    main()