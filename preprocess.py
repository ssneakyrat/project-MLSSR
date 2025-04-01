import os
import glob
import h5py
import numpy as np

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
            # Print all available keys
            print("Available datasets:", list(f.keys()))
            
            # Get the data
            phone_starts = f['PHONE_START'][:]
            phone_ends = f['PHONE_END'][:]
            phone_durations = f['PHONE_DURATION'][:]
            phone_texts = f['PHONE_TEXT'][:]
            file_name = f['FILE_NAME'][:]
            
            # Print file information
            print(f"FILE_NAME: {file_name[0]}")
            print(f"Number of phonemes: {len(phone_texts)}")
            
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

def save_to_h5py(bin_path, phonemes, file_path):
    """
    Save phoneme data to h5py binary file.
    
    Args:
        bin_path (str): Path to the binary file
        phonemes (list): List of phoneme tuples (start_time, end_time, phoneme)
        file_path (str): Original file path to save as FILE_NAME
    """
    # Create arrays for each component
    phone_starts = np.array([p[0] for p in phonemes])
    phone_ends = np.array([p[1] for p in phonemes])
    phone_durations = phone_ends - phone_starts
    phone_texts = np.array([p[2] for p in phonemes], dtype=h5py.special_dtype(vlen=str))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    
    # Create and save to h5py file
    with h5py.File(bin_path, 'w') as f:
        f.create_dataset('PHONE_START', data=phone_starts)
        f.create_dataset('PHONE_END', data=phone_ends)
        f.create_dataset('PHONE_DURATION', data=phone_durations)
        f.create_dataset('PHONE_TEXT', data=phone_texts)
        # Save the original file path as FILE_NAME
        f.create_dataset('FILE_NAME', data=np.array([file_path], dtype=h5py.special_dtype(vlen=str)))
    
    print(f"Saved phoneme data to {bin_path}")

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
        save_to_h5py(bin_path, phonemes, file_path)
        
        # Test reading the binary file
        test_read_h5py(bin_path)

if __name__ == "__main__":
    main()