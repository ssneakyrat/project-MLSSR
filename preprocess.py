import os
import glob
import h5py
import numpy as np
from tqdm import tqdm
import logging
import argparse

from utils.utils_general import load_config, plot_alignment
from utils.utils_dsp import extract_mel_spectrogram, extract_f0, f0_to_midi, estimate_phoneme_midi_notes
from utils.utils_transform import normalize_mel_spectrogram, pad_or_truncate_mel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger('preprocess')

def test_read_h5(h5_path):
    """
    Test reading from an H5 file and print its structure and contents.
    
    Args:
        h5_path (str): Path to the H5 file
    """
    logger.info(f"Testing read from H5 file: {h5_path}")
    try:
        with h5py.File(h5_path, 'r') as f:
            # Get dataset information
            dataset_keys = list(f.keys())
            
            logger.info(f"Dataset keys: {dataset_keys}")
            
            # Check for main data
            if 'mel_spectrograms' in f:
                data = f['mel_spectrograms']
                logger.info(f"Main data shape: {data.shape}")
                logger.info(f"Data type: {data.dtype}")
                
                # Print attributes
                logger.info("Data attributes:")
                for attr in data.attrs:
                    logger.info(f"  - {attr}: {data.attrs[attr]}")
                
                # Print sample values
                logger.info(f"Sample data (first item):")
                first_item = data[0]
                logger.info(f"  Shape: {first_item.shape}")
                logger.info(f"  Min: {np.min(first_item)}, Max: {np.max(first_item)}")
            
            # Check for metadata
            if 'file_ids' in f:
                file_ids = f['file_ids']
                logger.info(f"Number of file IDs: {len(file_ids)}")
                if len(file_ids) > 0:
                    logger.info(f"First 5 file IDs: {file_ids[:min(5, len(file_ids))]}")
            
            # Check for phoneme data
            if 'phoneme_count' in f:
                phoneme_count = f['phoneme_count']
                logger.info(f"Phoneme count statistics:")
                logger.info(f"  Average phonemes per file: {np.mean(phoneme_count):.2f}")
                logger.info(f"  Min phonemes: {np.min(phoneme_count)}")
                logger.info(f"  Max phonemes: {np.max(phoneme_count)}")
            
    except Exception as e:
        logger.error(f"Error reading H5 file: {e}")

def list_lab_files(raw_dir="raw_dir/lab"):
    """
    List all .lab files in the specified directory.
    """
    # Check if directory exists
    if not os.path.exists(raw_dir):
        logger.error(f"Error: {raw_dir} directory not found!")
        return []
    
    # Get all .lab files in the directory
    files = glob.glob(f"{raw_dir}/**/*.lab", recursive=True)
    
    # Print summary
    logger.info(f"Found {len(files)} .lab files in {raw_dir} directory")
    
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
        logger.error(f"Error parsing file {file_path}: {e}")
    
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
    
    logger.warning(f"Could not find corresponding .wav file for {lab_file_path}")
    return None

def save_to_h5(output_path, file_data, phone_map, config, data_key='mel_spectrograms', target_shape=None):
    """
    Save processed data to H5 file in a format optimized for training.
    
    Args:
        output_path (str): Path to the output H5 file
        file_data (dict): Dictionary containing processed data for each file
        phone_map (list): List of unique phonemes
        config (dict): Configuration dictionary
        data_key (str): Key to use for the data in the H5 file
        target_shape (tuple, optional): Target shape for mel spectrograms (mel_bins, time_frames)
                                      If None, uses shape from config.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"Saving processed data to H5 file: {output_path}")
    
    # Get target shape from config if not provided
    if target_shape is None:
        target_bins = config['model'].get('mel_bins', 80)
        target_frames = config['model'].get('time_frames', 128)
        target_shape = (target_bins, target_frames)
    
    # Count valid items first to create correctly sized datasets
    valid_items = 0
    for file_info in file_data.values():
        if 'MEL_SPEC' in file_info and file_info['MEL_SPEC'] is not None:
            valid_items += 1
    
    logger.info(f"Found {valid_items} valid items with mel spectrograms")
    
    # Create and save to h5py file
    with h5py.File(output_path, 'w') as f:
        # Convert phone_map to numpy array with proper dtype
        phone_map_array = np.array(phone_map, dtype=h5py.special_dtype(vlen=str))
        f.create_dataset('phone_map', data=phone_map_array)
        
        # Create the main dataset for mel spectrograms
        dataset = f.create_dataset(
            data_key,
            shape=(valid_items,) + target_shape,
            dtype=np.float32,
            chunks=(1,) + target_shape  # Chunk by individual spectrograms
        )
        
        # Create dataset for file IDs
        file_ids = f.create_dataset(
            'file_ids',
            shape=(valid_items,),
            dtype=h5py.special_dtype(vlen=str)
        )
        
        # Create datasets for phoneme information
        phoneme_count = f.create_dataset(
            'phoneme_count',
            shape=(valid_items,),
            dtype=np.int32
        )
        
        # Add metadata as attributes
        dataset.attrs['description'] = "Mel spectrograms extracted from audio files"
        dataset.attrs['shape_info'] = f"Shape: {target_shape} (frequency bins, time frames)"
        dataset.attrs['normalization'] = "Values are normalized to [0, 1] range"
        dataset.attrs['sample_rate'] = config['audio']['sample_rate']
        dataset.attrs['n_fft'] = config['audio']['n_fft']
        dataset.attrs['hop_length'] = config['audio']['hop_length']
        dataset.attrs['n_mels'] = config['audio']['n_mels']
        dataset.attrs['fmin'] = config['audio']['fmin']
        dataset.attrs['fmax'] = config['audio']['fmax']
        
        # Save data for each file with progress bar
        idx = 0
        with tqdm(total=len(file_data), desc="Saving to H5", unit="file") as pbar:
            for file_id, file_info in file_data.items():
                # Check if mel spectrogram exists
                if 'MEL_SPEC' in file_info and file_info['MEL_SPEC'] is not None:
                    # Get mel spectrogram
                    mel_spec = file_info['MEL_SPEC']
                    
                    # Normalize if needed
                    mel_spec = normalize_mel_spectrogram(mel_spec)
                    
                    # Pad or truncate to target dimensions
                    if mel_spec.shape != target_shape:
                        mel_spec = pad_or_truncate_mel(mel_spec, target_shape[1], target_shape[0])
                    
                    # Save to dataset
                    dataset[idx] = mel_spec
                    file_ids[idx] = file_id
                    
                    # Save phoneme count if available
                    if 'PHONE_TEXT' in file_info:
                        phoneme_count[idx] = len(file_info['PHONE_TEXT'])
                    else:
                        phoneme_count[idx] = 0
                    
                    idx += 1
                
                pbar.update(1)
    
        # Create a visualization group to store sample plots
        if idx > 0:
            # Select a few random samples for visualization
            viz_samples = min(3, idx)
            random_indices = np.random.choice(idx, viz_samples, replace=False)
            
            # Create visualization plots for selected samples
            viz_group = f.create_group('visualizations')
            
            for i, sample_idx in enumerate(random_indices):
                try:
                    # Get sample info
                    sample_id = file_ids[sample_idx]
                    file_info = file_data[sample_id]
                    
                    # Get phoneme tuples for visualization
                    if 'PHONE_TEXT' in file_info and 'PHONE_START' in file_info and 'PHONE_END' in file_info:
                        phone_tuples = []
                        for j in range(len(file_info['PHONE_TEXT'])):
                            phone_tuples.append((
                                file_info['PHONE_START'][j],
                                file_info['PHONE_END'][j],
                                file_info['PHONE_TEXT'][j]
                            ))
                        
                        # Create phoneme MIDI tuples if available
                        phoneme_midi_tuples = None
                        if 'PHONEME_MIDI' in file_info:
                            phoneme_midi_tuples = []
                            for j in range(len(file_info['PHONE_TEXT'])):
                                phoneme_midi_tuples.append((
                                    file_info['PHONE_START'][j],
                                    file_info['PHONE_END'][j],
                                    file_info['PHONE_TEXT'][j],
                                    file_info['PHONEME_MIDI'][j]
                                ))
                        
                        # Store the original mel spectrogram and F0 for visualization plots
                        if 'F0' in file_info and file_info['F0'] is not None:
                            # Generate and store the alignment visualization as embedded image
                            # (not implemented in this version but could be added)
                            viz_group.create_dataset(f'sample_{i}_id', data=np.string_(sample_id))
                            viz_group.create_dataset(f'sample_{i}_phoneme_count', 
                                                    data=len(file_info['PHONE_TEXT']))
                    
                except Exception as e:
                    logger.error(f"Error creating visualization for sample {sample_idx}: {e}")
    
    logger.info(f"Saved {idx} mel spectrograms to {output_path}")
    logger.info(f"Dataset shape: {target_shape}")
    
    # Test reading the H5 file
    test_read_h5(output_path)

def collect_unique_phonemes(lab_files):
    """
    Collect unique phonemes from all lab files.
    
    Args:
        lab_files (list): List of lab file paths
    
    Returns:
        list: List of unique phonemes
    """
    unique_phonemes = set()
    
    # Use progress bar for collecting phonemes
    with tqdm(total=len(lab_files), desc="Collecting unique phonemes", unit="file") as pbar:
        for file_path in lab_files:
            phonemes = parse_lab_file(file_path)
            for _, _, phone in phonemes:
                unique_phonemes.add(phone)
            pbar.update(1)
    
    # Convert to sorted list for consistent indexing
    phone_map = sorted(list(unique_phonemes))
    logger.info(f"Collected {len(phone_map)} unique phonemes: {phone_map}")
    
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
    logger.info("Updated configuration with phone map")

def main():
    """
    Main function to process lab files and save data to H5 file.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process lab files and save data to H5 file')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--raw_dir', type=str,
                        help='Raw directory path (overrides config)')
    parser.add_argument('--output', type=str,
                        help='Path for the output H5 file (overrides config)')
    parser.add_argument('--min_phonemes', type=int, default=5,
                        help='Minimum number of phonemes required per file')
    parser.add_argument('--data_key', type=str, default='mel_spectrograms',
                        help='Key to use for data in the H5 file')
    parser.add_argument('--target_length', type=int, default=None,
                        help='Target number of time frames for mel spectrograms')
    parser.add_argument('--target_bins', type=int, default=None,
                        help='Target number of mel bins for mel spectrograms')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments if provided
    if args.raw_dir:
        config['data']['raw_dir'] = args.raw_dir
    
    # Get paths from config
    raw_dir = config['data']['raw_dir']
    
    # Determine output path
    output_path = args.output
    if output_path is None:
        # Use bin_dir and filename from config
        bin_dir = config['data']['bin_dir']
        bin_file = config['data']['bin_file']
        output_path = os.path.join(bin_dir, bin_file)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create plots directory if it doesn't exist
    plots_dir = config.get('audio', {}).get('plots_dir', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get target shape for mel spectrograms
    target_shape = None
    if args.target_length is not None and args.target_bins is not None:
        target_shape = (args.target_bins, args.target_length)
    
    lab_files = list_lab_files(raw_dir)
    
    # Collect unique phonemes from all files
    phone_map = collect_unique_phonemes(lab_files)
    
    # Update the configuration with the phone map
    update_config_with_phone_map(config, phone_map)
    
    # Dictionary to store all processed data
    all_file_data = {}
    
    # Keep track of files skipped due to minimum phoneme requirement
    min_phoneme_count = args.min_phonemes
    skipped_files_count = 0
    processed_files_count = 0
    
    # Process each file with progress bar
    logger.info("Processing lab files...")
    with tqdm(total=len(lab_files), desc="Processing files", unit="file") as pbar:
        for file_path in lab_files:
            # Parse the lab file
            phonemes = parse_lab_file(file_path)
            
            # Skip files with fewer than minimum required phonemes
            if len(phonemes) < min_phoneme_count:
                tqdm.write(f"Skipping {os.path.basename(file_path)}: Only has {len(phonemes)} phonemes (min: {min_phoneme_count})")
                skipped_files_count += 1
                pbar.update(1)
                continue
            
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
            
            processed_files_count += 1
            pbar.update(1)
    
    # Print summary of files processed and skipped
    logger.info(f"\nSummary:")
    logger.info(f"Total files found: {len(lab_files)}")
    logger.info(f"Files processed: {processed_files_count}")
    logger.info(f"Files skipped (fewer than {min_phoneme_count} phonemes): {skipped_files_count}")
    
    # Save data to H5 file
    if all_file_data:
        save_to_h5(output_path, all_file_data, phone_map, config, args.data_key, target_shape)
    else:
        logger.warning("No files were processed. H5 file was not created.")

if __name__ == "__main__":
    main()