import os
import glob
import h5py
import numpy as np
import yaml

from utils.utils_general import load_config

def test_read_h5py(bin_path):
    """
    Test reading from an h5py binary file and print its contents.
    
    Args:
        bin_path (str): Path to the binary file
    """
    print(f"\nTesting read from {bin_path}:")
    try:
        with h5py.File(bin_path, 'r') as f:
            # Print all available keys at root level
            print("Root level keys:", list(f.keys()))
            
            # Get the phone map at root level
            phone_map = f['PHONE_MAP'][:]
            print(f"Phone map: {[p.decode('utf-8') if isinstance(p, bytes) else p for p in phone_map]}")
            
            # Get data from the data group
            data_group = f['data']
            print("Data group keys:", list(data_group.keys()))
            
            # Get the data
            phone_starts = data_group['PHONE_START'][:]
            phone_ends = data_group['PHONE_END'][:]
            phone_durations = data_group['PHONE_DURATION'][:]
            phone_texts = data_group['PHONE_TEXT'][:]
            file_name = data_group['FILE_NAME'][:]
            total_duration = data_group['TOTAL_DURATION'][:]
            
            # Print file information
            print(f"FILE_NAME: {file_name[0]}")
            print(f"Number of phonemes: {len(phone_texts)}")
            print(f"Total duration: {total_duration[0]}")
            
            # Print first 5 entries
            print("First 5 entries:")
            for i in range(min(5, len(phone_texts))):
                print(f"  {phone_texts[i]}: start={phone_starts[i]}, end={phone_ends[i]}, duration={phone_durations[i]}")
    except Exception as e:
        print(f"Error reading h5py file: {e}")
        
def list_lab_files(raw_dir="raw_dir/lab"):
    """
    List all .lab files in the specified directory.
    """
    # Check if directory exists
    if not os.path.exists(raw_dir):
        print(f"Error: {raw_dir} directory not found!")
        return []
    
    # Get all .lab files in the directory
    files = glob.glob(f"{raw_dir}/**/*.lab", recursive=True)
    
    # Print the list of files
    print(f"Found {len(files)} .lab files in {raw_dir} directory:")
    for file in files:
        print(f"  - {file}")
    
    return files

def parse_lab_file(file_path):
    """
    Parse a .lab file containing phoneme timing data.
    """
    phonemes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Split line into components
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time = int(parts[0])
                    end_time = int(parts[1])
                    phoneme = parts[2]
                    phonemes.append((start_time, end_time, phoneme))
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
    
    return phonemes

def save_to_h5py(bin_path, phonemes, file_path, phone_map):
    """
    Save phoneme data to h5py binary file.
    
    Args:
        bin_path (str): Path to the binary file
        phonemes (list): List of phoneme tuples (start_time, end_time, phoneme)
        file_path (str): Original file path to save as FILE_NAME
        phone_map (list): List of unique phonemes
    """
    # Create arrays for each component
    phone_starts = np.array([p[0] for p in phonemes])
    phone_ends = np.array([p[1] for p in phonemes])
    phone_durations = phone_ends - phone_starts
    phone_texts = np.array([p[2] for p in phonemes], dtype=h5py.special_dtype(vlen=str))
    
    # Calculate total duration by summing all phone durations
    total_duration = np.sum(phone_durations)
    
    # Convert phone_map to numpy array with proper dtype
    phone_map_array = np.array(phone_map, dtype=h5py.special_dtype(vlen=str))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    
    # Create and save to h5py file
    with h5py.File(bin_path, 'w') as f:
        # Save the phone map at root level
        f.create_dataset('PHONE_MAP', data=phone_map_array)
        
        # Create a data group for all other information
        data_group = f.create_group('data')
        
        # Save phoneme data in the data group
        data_group.create_dataset('PHONE_START', data=phone_starts)
        data_group.create_dataset('PHONE_END', data=phone_ends)
        data_group.create_dataset('PHONE_DURATION', data=phone_durations)
        data_group.create_dataset('PHONE_TEXT', data=phone_texts)
        
        # Save the original file path as FILE_NAME in the data group
        data_group.create_dataset('FILE_NAME', data=np.array([file_path], dtype=h5py.special_dtype(vlen=str)))
        
        # Save the total duration in the data group
        data_group.create_dataset('TOTAL_DURATION', data=np.array([total_duration]))
    
    print(f"Saved phoneme data to {bin_path}")
    print(f"Total duration: {total_duration}")
    print(f"Phone map: {phone_map}")

def collect_unique_phonemes(lab_files):
    """
    Collect unique phonemes from all lab files.
    
    Args:
        lab_files (list): List of lab file paths
    
    Returns:
        list: List of unique phonemes
    """
    unique_phonemes = set()
    
    for file_path in lab_files:
        phonemes = parse_lab_file(file_path)
        for _, _, phone in phonemes:
            unique_phonemes.add(phone)
    
    # Convert to sorted list for consistent indexing
    phone_map = sorted(list(unique_phonemes))
    print(f"Collected {len(phone_map)} unique phonemes: {phone_map}")
    
    return phone_map

def update_config_with_phone_map(config, phone_map, config_path="config/default.yaml"):
    """
    Update the configuration file with the phone map.
    
    Args:
        config (dict): Configuration dictionary
        phone_map (list): List of unique phonemes
        config_path (str): Path to the configuration file
    """
    # Update the phone_map in the config
    config['data']['phone_map'] = phone_map
    
    # Write back to the config file
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Updated configuration file {config_path} with phone map")
    except Exception as e:
        print(f"Error updating configuration file: {e}")

def main():
    """
    Main function to process lab files.
    """
    # Load configuration
    config = load_config()
    
    # Get list of .lab files
    raw_dir = config['data']['raw_dir']
    bin_dir = config['data']['bin_dir']
    
    # Ensure bin directory exists
    os.makedirs(bin_dir, exist_ok=True)
    
    lab_files = list_lab_files(raw_dir)
    
    # Collect unique phonemes from all files
    phone_map = collect_unique_phonemes(lab_files)
    
    # Update the configuration with the phone map
    update_config_with_phone_map(config, phone_map)
    
    # Process each file
    for file_path in lab_files:
        print(f"\nProcessing file: {file_path}")
        phonemes = parse_lab_file(file_path)
        
        # Print phoneme count
        print(f"Found {len(phonemes)} phonemes:")
        
        # Generate bin filename
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        bin_path = os.path.join(bin_dir, f"{base_filename}.h5")
        
        # Save to h5py binary file
        save_to_h5py(bin_path, phonemes, file_path, phone_map)
        
        # Test reading the binary file
        test_read_h5py(bin_path)

if __name__ == "__main__":
    main()