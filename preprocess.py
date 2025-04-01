import os
import glob
import h5py
import numpy as np

from utils.utils_general import load_config, plot_alignment
from utils.utils_dsp import extract_mel_spectrogram, extract_f0, f0_to_midi, estimate_phoneme_midi_notes

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
            
            # Get all file IDs
            file_ids = list(data_group.keys())
            
            # Print number of files
            print(f"Number of files in dataset: {len(file_ids)}")
            
            # Select a random file to visualize
            if file_ids:
                import random
                file_id = random.choice(file_ids)
                print(f"\nVisualizing random file: {file_id}")
                
                file_group = data_group[file_id]
                
                # Get the data for the selected file
                phone_starts = file_group['PHONE_START'][:]
                phone_ends = file_group['PHONE_END'][:]
                phone_durations = file_group['PHONE_DURATION'][:]
                phone_texts = file_group['PHONE_TEXT'][:]
                file_name = file_group['FILE_NAME'][:]
                total_duration = file_group['TOTAL_DURATION'][:]
                
                # Print file information
                print(f"FILE_NAME: {file_name[0]}")
                print(f"Number of phonemes: {len(phone_texts)}")
                print(f"Total duration: {total_duration[0]}")
                
                # Print first 5 entries
                print("First 5 entries:")
                for i in range(min(5, len(phone_texts))):
                    print(f"  {phone_texts[i]}: start={phone_starts[i]}, end={phone_ends[i]}, duration={phone_durations[i]}")
                
                # Check if mel spectrogram, F0, and MIDI_NOTES are present
                mel_spec = None
                f0 = None
                midi_notes = None
                
                if 'MEL_SPEC' in file_group:
                    mel_spec = file_group['MEL_SPEC'][:]
                    print(f"MEL_SPEC shape: {mel_spec.shape}")
                    
                if 'F0' in file_group:
                    f0 = file_group['F0'][:]
                    print(f"F0 shape: {f0.shape}")
                    
                if 'MIDI_NOTES' in file_group:
                    midi_notes = file_group['MIDI_NOTES'][:]
                    print(f"MIDI_NOTES shape: {midi_notes.shape}")
                    
                if 'PHONEME_MIDI' in file_group:
                    phoneme_midi = file_group['PHONEME_MIDI'][:]
                    print(f"PHONEME_MIDI (first 5 entries):")
                    for i in range(min(5, len(phoneme_midi))):
                        print(f"  {phone_texts[i]}: MIDI note = {phoneme_midi[i]}")
                    
                # Generate visualization if both mel and F0 are available
                if mel_spec is not None and f0 is not None:
                    # Create plots directory if it doesn't exist
                    plots_dir = config.get('audio', {}).get('plots_dir', 'plots')
                    os.makedirs(plots_dir, exist_ok=True)
                    
                    # Create phoneme tuples for alignment visualization
                    phoneme_tuples = []
                    phoneme_midi_tuples = []
                    for i in range(len(phone_texts)):
                        phoneme_tuples.append((phone_starts[i], phone_ends[i], phone_texts[i]))
                        if 'PHONEME_MIDI' in file_group:
                            phoneme_midi_tuples.append((phone_starts[i], phone_ends[i], phone_texts[i], phoneme_midi[i]))
                        else:
                            phoneme_midi_tuples = None
                    
                    # Generate and save the alignment visualization
                    alignment_plot_path = os.path.join(plots_dir, f"{file_id}_alignment.png")
                    plot_alignment(
                        mel_spec, 
                        f0, 
                        phoneme_tuples, 
                        alignment_plot_path, 
                        config, 
                        phoneme_midi=phoneme_midi_tuples,
                        midi_notes=midi_notes
                    )
                
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

def save_all_to_h5py(bin_path, file_data, phone_map):
    """
    Save all processed data to a single h5py binary file.
    
    Args:
        bin_path (str): Path to the binary file
        file_data (dict): Dictionary containing processed data for each file
        phone_map (list): List of unique phonemes
    """
    # Convert phone_map to numpy array with proper dtype
    phone_map_array = np.array(phone_map, dtype=h5py.special_dtype(vlen=str))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    
    # Create and save to h5py file
    with h5py.File(bin_path, 'w') as f:
        # Save the phone map at root level
        f.create_dataset('PHONE_MAP', data=phone_map_array)
        
        # Create a data group for all files
        data_group = f.create_group('data')
        
        # Save data for each file
        for file_id, file_info in file_data.items():
            # Create a group for this file
            file_group = data_group.create_group(file_id)
            
            # Save phoneme data
            file_group.create_dataset('PHONE_START', data=file_info['PHONE_START'])
            file_group.create_dataset('PHONE_END', data=file_info['PHONE_END'])
            file_group.create_dataset('PHONE_DURATION', data=file_info['PHONE_DURATION'])
            file_group.create_dataset('PHONE_TEXT', data=file_info['PHONE_TEXT'])
            file_group.create_dataset('FILE_NAME', data=file_info['FILE_NAME'])
            file_group.create_dataset('TOTAL_DURATION', data=file_info['TOTAL_DURATION'])
            
            # Save mel spectrogram if available
            if 'MEL_SPEC' in file_info and file_info['MEL_SPEC'] is not None:
                file_group.create_dataset('MEL_SPEC', data=file_info['MEL_SPEC'])
            
            # Save F0 if available
            if 'F0' in file_info and file_info['F0'] is not None:
                file_group.create_dataset('F0', data=file_info['F0'])
            
            # Save MIDI notes if available
            if 'MIDI_NOTES' in file_info and file_info['MIDI_NOTES'] is not None:
                file_group.create_dataset('MIDI_NOTES', data=file_info['MIDI_NOTES'])
            
            # Save phoneme MIDI notes if available
            if 'PHONEME_MIDI' in file_info and file_info['PHONEME_MIDI'] is not None:
                file_group.create_dataset('PHONEME_MIDI', data=file_info['PHONEME_MIDI'])
    
    print(f"Saved all data to single binary file: {bin_path}")
    print(f"Total number of files saved: {len(file_data)}")

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
    Main function to process lab files and save all data to a single h5py file.
    """
    # Load configuration
    config = load_config()
    
    # Get list of .lab files
    raw_dir = config['data']['raw_dir']
    bin_dir = config['data']['bin_dir']
    bin_file = config['data']['bin_file']
    
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
    
    # Dictionary to store all processed data
    all_file_data = {}
    
    # Process each file
    for file_path in lab_files:
        print(f"\nProcessing file: {file_path}")
        phonemes = parse_lab_file(file_path)
        
        # Find the corresponding wav file
        wav_file_path = find_wav_file(file_path, raw_dir)
        
        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        file_id = base_filename
        
        # Extract mel spectrogram, F0, and calculate MIDI notes if wav file found
        mel_spec = None
        f0 = None
        midi_notes = None
        phoneme_midi = None
        if wav_file_path:
            print(f"Found corresponding .wav file: {wav_file_path}")
            
            # Extract mel spectrogram
            mel_spec = extract_mel_spectrogram(wav_file_path, config)
            
            # Extract F0 contour
            f0 = extract_f0(wav_file_path, config)
            
            # Convert F0 to MIDI notes
            if f0 is not None:
                midi_notes = f0_to_midi(f0)
                
                # Estimate MIDI notes for each phoneme
                sample_rate = config['audio']['sample_rate']
                hop_length = config['audio']['hop_length']
                phoneme_midi_tuples = estimate_phoneme_midi_notes(
                    f0, 
                    phonemes, 
                    hop_length, 
                    sample_rate
                )
                
                # Convert to numpy array
                phoneme_midi = np.array([midi for _, midi in phoneme_midi_tuples])
                
                print(f"Estimated MIDI notes for {len(phoneme_midi_tuples)} phonemes")
        else:
            print(f"Warning: No corresponding .wav file found for {file_path}")
        
        # Create arrays for each component
        phone_starts = np.array([p[0] for p in phonemes])
        phone_ends = np.array([p[1] for p in phonemes])
        phone_durations = phone_ends - phone_starts
        phone_texts = np.array([p[2] for p in phonemes], dtype=h5py.special_dtype(vlen=str))
        
        # Calculate total duration by summing all phone durations
        total_duration = np.sum(phone_durations)
        
        # Store the data for this file
        all_file_data[file_id] = {
            'PHONE_START': phone_starts,
            'PHONE_END': phone_ends,
            'PHONE_DURATION': phone_durations,
            'PHONE_TEXT': phone_texts,
            'FILE_NAME': np.array([file_path], dtype=h5py.special_dtype(vlen=str)),
            'TOTAL_DURATION': np.array([total_duration]),
            'MEL_SPEC': mel_spec,
            'F0': f0,
            'MIDI_NOTES': midi_notes,
            'PHONEME_MIDI': phoneme_midi
        }
    
    # Save all data to a single binary file
    bin_path = os.path.join(bin_dir, bin_file)
    save_all_to_h5py(bin_path, all_file_data, phone_map)
    
    # Test reading the binary file and generate visualization
    test_read_h5py(bin_path, config)

if __name__ == "__main__":
    main()