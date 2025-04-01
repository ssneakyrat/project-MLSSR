import h5py
import os
import random
from utils.utils_general import load_config

def test_load_data():
    """
    Test function to load data from a single h5 binary file and print 3 random sample file names.
    
    Returns:
        list: List of file names for the 3 random samples
    """
    # Load configuration
    config = load_config()
    
    # Get the path to the binary directory and file
    bin_dir = config['data']['bin_dir']
    bin_file = config['data']['bin_file']
    bin_path = os.path.join(bin_dir, bin_file)
    
    print(f"Loading data from {bin_path}")
    
    # Check if the binary file exists
    if not os.path.exists(bin_path):
        print(f"Error: Binary file {bin_path} not found!")
        return []
    
    try:
        with h5py.File(bin_path, 'r') as f:
            # Check if there's a data group
            if 'data' in f:
                data_group = f['data']
                
                # Get all file IDs (keys in the data group)
                file_ids = list(data_group.keys())
                
                if not file_ids:
                    print("Error: No files found in the data group")
                    return []
                
                # Print total number of files
                print(f"Total number of files in dataset: {len(file_ids)}")
                
                # Select 3 random files or fewer if there are less than 3 samples
                num_samples = min(3, len(file_ids))
                random_ids = random.sample(file_ids, num_samples)
                
                # Get the file names for the selected files
                selected_files = []
                
                print(f"Selected {num_samples} random samples:")
                for i, file_id in enumerate(random_ids):
                    file_group = data_group[file_id]
                    
                    if 'FILE_NAME' in file_group:
                        file_name = file_group['FILE_NAME'][0]
                        selected_files.append(file_name)
                        print(f"  {i+1}. ID: {file_id}, File: {file_name}")
                        
                        # Print additional information
                        if 'PHONE_TEXT' in file_group:
                            num_phonemes = len(file_group['PHONE_TEXT'])
                            print(f"     Number of phonemes: {num_phonemes}")
                        
                        if 'MEL_SPEC' in file_group:
                            mel_shape = file_group['MEL_SPEC'].shape
                            print(f"     Mel spectrogram shape: {mel_shape}")
                    else:
                        print(f"  {i+1}. ID: {file_id} - FILE_NAME not found")
                
                return selected_files
            else:
                print("Error: 'data' group not found in the binary file")
    except Exception as e:
        print(f"Error loading data from binary file: {e}")
    
    return []

def load_dataset(split="train", shuffle=True):
    """
    Load the dataset for training or evaluation.
    
    Args:
        split (str): Dataset split, either "train", "val", or "test"
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        list: List of data items
    """
    # Load configuration
    config = load_config()
    
    # Get the path to the binary directory and file
    bin_dir = config['data']['bin_dir']
    bin_file = config['data']['bin_file']
    bin_path = os.path.join(bin_dir, bin_file)
    
    # Check if the binary file exists
    if not os.path.exists(bin_path):
        print(f"Error: Binary file {bin_path} not found!")
        return []
    
    # Load all data
    data_items = []
    
    try:
        with h5py.File(bin_path, 'r') as f:
            if 'data' in f:
                data_group = f['data']
                
                # Get all file IDs
                file_ids = list(data_group.keys())
                
                # Print total number of files
                print(f"Loading {split} split from {len(file_ids)} total files")
                
                # For this example, use all data for all splits
                # In a real implementation, you would divide the data based on the split
                
                # Process each file
                for file_id in file_ids:
                    file_group = data_group[file_id]
                    
                    # Create a data item
                    data_item = {
                        'id': file_id,
                        'file_name': file_group['FILE_NAME'][0]
                    }
                    
                    # Add phoneme data
                    if 'PHONE_TEXT' in file_group:
                        data_item['phone_texts'] = file_group['PHONE_TEXT'][:]
                        data_item['phone_starts'] = file_group['PHONE_START'][:]
                        data_item['phone_ends'] = file_group['PHONE_END'][:]
                        data_item['phone_durations'] = file_group['PHONE_DURATION'][:]
                    
                    # Add mel spectrogram
                    if 'MEL_SPEC' in file_group:
                        data_item['mel_spec'] = file_group['MEL_SPEC'][:]
                    
                    # Add F0
                    if 'F0' in file_group:
                        data_item['f0'] = file_group['F0'][:]
                    
                    # Add MIDI notes
                    if 'MIDI_NOTES' in file_group:
                        data_item['midi_notes'] = file_group['MIDI_NOTES'][:]
                    
                    # Add phoneme MIDI
                    if 'PHONEME_MIDI' in file_group:
                        data_item['phoneme_midi'] = file_group['PHONEME_MIDI'][:]
                    
                    # Add the data item to the list
                    data_items.append(data_item)
    except Exception as e:
        print(f"Error loading dataset: {e}")
    
    # Shuffle the data if requested
    if shuffle:
        random.shuffle(data_items)
    
    print(f"Loaded {len(data_items)} items for {split} split")
    return data_items