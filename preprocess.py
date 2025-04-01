import os
import glob
import h5py
import numpy as np
import yaml

from utils.utils_general import load_config
from utils.utils_dsp import extract_mel_spectrogram

def test_read_h5py(bin_path, config):
    """
    Test reading from an h5py binary file, print its contents, and generate visualization.
    
    Args:
        bin_path (str): Path to the binary file
        config (dict): Configuration dictionary
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
            
            # Check if mel spectrogram and F0 are present
            mel_spec = None
            f0 = None
            
            if 'MEL_SPEC' in data_group:
                mel_spec = data_group['MEL_SPEC'][:]
                print(f"MEL_SPEC shape: {mel_spec.shape}")
                
            if 'F0' in data_group:
                f0 = data_group['F0'][:]
                print(f"F0 shape: {f0.shape}")
                
            # Generate visualization if both mel and F0 are available
            if mel_spec is not None and f0 is not None:
                # Create plots directory if it doesn't exist
                plots_dir = config.get('audio', {}).get('plots_dir', 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                
                # Get base filename without extension
                base_filename = os.path.splitext(os.path.basename(bin_path))[0]
                plot_path = os.path.join(plots_dir, f"{base_filename}_visualization.png")
                
                # Import the plotting function from utils_dsp
                from utils.utils_dsp import plot_mel_and_f0
                
                # Plot and save
                plot_mel_and_f0(mel_spec, f0, plot_path)
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

def find_wav_file(lab_file_path, raw_dir):
    """
    Find the corresponding .wav file for a .lab file.
    
    Args:
        lab_file_path (str): Path to the .lab file
        raw_dir (str): Raw directory path
        
    Returns:
        str: Path to the corresponding .wav file, or None if not found
    """
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(lab_file_path))[0]
    
    # Get the directory of the lab file
    lab_dir = os.path.dirname(lab_file_path)
    
    # Try to find wav file in parallel 'wav' directory
    wav_dir = lab_dir.replace('/lab/', '/wav/')
    if '/lab/' not in wav_dir:
        wav_dir = lab_dir.replace('\\lab\\', '\\wav\\')
    
    wav_file_path = os.path.join(wav_dir, f"{base_filename}.wav")
    
    # Check if the wav file exists
    if os.path.exists(wav_file_path):
        return wav_file_path
    
    # If not found, try to find it in the raw directory's wav folder
    wav_file_path = os.path.join(raw_dir, "wav", f"{base_filename}.wav")
    
    if os.path.exists(wav_file_path):
        return wav_file_path
    
    print(f"Warning: Could not find corresponding .wav file for {lab_file_path}")
    return None

def save_to_h5py(bin_path, phonemes, file_path, phone_map, mel_spec=None, f0=None):
    """
    Save phoneme data, mel spectrogram, and F0 to h5py binary file.
    
    Args:
        bin_path (str): Path to the binary file
        phonemes (list): List of phoneme tuples (start_time, end_time, phoneme)
        file_path (str): Original file path to save as FILE_NAME
        phone_map (list): List of unique phonemes
        mel_spec (numpy.ndarray, optional): Mel spectrogram
        f0 (numpy.ndarray, optional): F0 contour
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
        
        # Save the mel spectrogram if provided
        if mel_spec is not None:
            data_group.create_dataset('MEL_SPEC', data=mel_spec)
            
        # Save the F0 contour if provided
        if f0 is not None:
            data_group.create_dataset('F0', data=f0)
    
    '''
    print(f"Saved phoneme data to {bin_path}")
    print(f"Total duration: {total_duration}")
    if mel_spec is not None:
        print(f"Mel spectrogram shape: {mel_spec.shape}")
    if f0 is not None:
        print(f"F0 contour shape: {f0.shape}")'
    '''

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
    
    # Create plots directory if it doesn't exist
    plots_dir = config.get('audio', {}).get('plots_dir', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
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
        #print(f"Found {len(phonemes)} phonemes")
        
        # Generate bin filename
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        bin_path = os.path.join(bin_dir, f"{base_filename}.h5")
        
        # Find the corresponding wav file
        wav_file_path = find_wav_file(file_path, raw_dir)
        
        # Extract mel spectrogram and F0 if wav file found
        mel_spec = None
        f0 = None
        if wav_file_path:
            #print(f"Found corresponding .wav file: {wav_file_path}")
            
            # Extract mel spectrogram
            mel_spec = extract_mel_spectrogram(wav_file_path, config)
            
            # Extract F0 contour
            from utils.utils_dsp import extract_f0
            f0 = extract_f0(wav_file_path, config)
        else:
            print(f"Warning: No corresponding .wav file found for {file_path}")
        
        # Save to h5py binary file
        save_to_h5py(bin_path, phonemes, file_path, phone_map, mel_spec, f0)
        
        # Test reading the binary file and generate visualization
        test_read_h5py(bin_path, config)

if __name__ == "__main__":
    main()