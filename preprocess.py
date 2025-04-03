import os
import glob
import h5py
import numpy as np
import math
from tqdm import tqdm
import argparse
import logging
import warnings

from utils.utils import (
    load_config, 
    extract_mel_spectrogram,
    extract_mel_spectrogram_variable_length,
    extract_f0, 
    extract_aligned_f0,
    normalize_mel_spectrogram, 
    pad_or_truncate_mel
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('preprocess')

def f0_to_midi_pitch(f0, silence_threshold=20):
    """Convert f0 frequency in Hz to MIDI pitch number.
    MIDI pitch 69 is A4 (440 Hz).
    Each semitone is a step of 1 in MIDI pitch.
    
    Args:
        f0: Fundamental frequency in Hz
        silence_threshold: Values below this threshold are considered silence/unvoiced
        
    Returns:
        MIDI pitch number (float) or 0 for silence/unvoiced
    """
    if f0 is None or f0 <= silence_threshold:
        return 0  # For unvoiced or silent regions
    return 69 + 12 * np.log2(f0 / 440.0)

def is_midi_pitch(f0_array):
    """Detect if an array is already in MIDI pitch format rather than Hz.
    
    Args:
        f0_array: Array of F0 values
        
    Returns:
        Boolean indicating if the array appears to be in MIDI format
    """
    # MIDI pitch for human voice is typically between 36-84 (C2-C6)
    # F0 for human voice is typically between 80-1000 Hz
    
    # Get only the voiced values (non-zero)
    voiced = f0_array[f0_array > 0]
    
    if len(voiced) == 0:
        return False
    
    mean_value = np.mean(voiced)
    max_value = np.max(voiced)
    
    # If mean is less than 100 and max is less than 120, likely MIDI
    # If mean is greater than 100, likely Hz
    return mean_value < 100 and max_value < 120

def estimate_phoneme_midi_pitch(f0, phoneme_start, phoneme_end, hop_length, sample_rate):
    """Estimate the average MIDI pitch for a phoneme with improved handling of formats.
    
    Args:
        f0: Array of F0 values (aligned with mel spectrogram frames)
        phoneme_start: Start time in samples
        phoneme_end: End time in samples
        hop_length: Hop length used for F0 extraction
        sample_rate: Sample rate of the audio
        
    Returns:
        Average MIDI pitch for the phoneme
    """
    if f0 is None or len(f0) == 0:
        return 0
    
    # First, detect if f0 is already in MIDI format
    is_midi = is_midi_pitch(f0)
    
    # Convert time in samples to frame indices correctly
    start_frame = int(np.floor(phoneme_start / hop_length))
    end_frame = int(np.ceil(phoneme_end / hop_length))
    
    # Ensure bounds are within the F0 array
    start_frame = max(0, start_frame)
    end_frame = min(len(f0), end_frame)
    
    # Make sure we have at least one frame
    if start_frame >= end_frame or start_frame >= len(f0):
        return 0
    
    # Extract F0 values for the phoneme duration
    phoneme_f0 = f0[start_frame:end_frame]
    
    # Filter out zeros and very low values (unvoiced or silent regions)
    if is_midi:
        # For MIDI data, just filter out zeros
        voiced_f0 = phoneme_f0[phoneme_f0 > 0]
    else:
        # For Hz data, use a threshold of 30Hz to filter out noise
        voiced_f0 = phoneme_f0[phoneme_f0 > 30]
    
    # If no voiced frames are found, return 0
    if len(voiced_f0) == 0:
        return 0
    
    # Calculate average F0 for the phoneme
    avg_f0 = float(np.mean(voiced_f0))
    
    # Convert to MIDI pitch if needed
    if is_midi:
        # Already in MIDI format, just return as is
        return avg_f0
    else:
        # Convert from Hz to MIDI
        return f0_to_midi_pitch(avg_f0)

def list_lab_files(raw_dir):
    if not os.path.exists(raw_dir):
        logger.error(f"Error: {raw_dir} directory not found!")
        return []
    
    files = glob.glob(f"{raw_dir}/**/*.lab", recursive=True)
    logger.info(f"Found {len(files)} .lab files in {raw_dir} directory")
    
    return files

def parse_lab_file(file_path):
    phonemes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
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
    base_filename = os.path.splitext(os.path.basename(lab_file_path))[0]
    lab_dir = os.path.dirname(lab_file_path)
    
    wav_dir = lab_dir.replace('/lab/', '/wav/')
    if '/lab/' not in wav_dir:
        wav_dir = lab_dir.replace('\\lab\\', '\\wav\\')
    
    wav_file_path = os.path.join(wav_dir, f"{base_filename}.wav")
    
    if os.path.exists(wav_file_path):
        return wav_file_path
    
    wav_file_path = os.path.join(raw_dir, "wav", f"{base_filename}.wav")
    
    if os.path.exists(wav_file_path):
        return wav_file_path
    
    return None

def save_to_h5_variable_length(output_path, file_data, phone_map, config, data_key='mel_spectrograms'):
    """Save mel spectrograms to H5 file with variable length support and aligned F0"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mel_bins = config['model'].get('mel_bins', 80)
    
    # Calculate max possible time frames for the maximum audio length
    max_audio_length = config['audio'].get('max_audio_length', 10.0)  # Default 10 seconds
    sample_rate = config['audio'].get('sample_rate', 22050)
    hop_length = config['audio'].get('hop_length', 256)
    max_time_frames = math.ceil(max_audio_length * sample_rate / hop_length)
    
    # Count valid items and get the maximum length
    valid_items = 0
    max_length = 0
    for file_info in file_data.values():
        if 'MEL_SPEC' in file_info and file_info['MEL_SPEC'] is not None:
            valid_items += 1
            curr_length = file_info['MEL_SPEC'].shape[1]  # Time dimension
            max_length = max(max_length, curr_length)
    
    # Cap the maximum length to max_time_frames
    max_length = min(max_length, max_time_frames)
    logger.info(f"Maximum time frames: {max_length} (equivalent to {max_length * hop_length / sample_rate:.2f} seconds)")
    
    with h5py.File(output_path, 'w') as f:
        # Store phone map
        phone_map_array = np.array(phone_map, dtype=h5py.special_dtype(vlen=str))
        f.create_dataset('phone_map', data=phone_map_array)
        
        # Create a ragged dataset to store variable-length mel spectrograms
        dataset = f.create_dataset(
            data_key,
            shape=(valid_items, mel_bins, max_length),
            dtype=np.float32,
            chunks=(1, mel_bins, min(128, max_length))  # Chunk size for efficient access
        )
        
        # Create a dataset for F0 data with the same time dimension as mel spectrograms
        f0_dataset = f.create_dataset(
            'f0_aligned',
            shape=(valid_items, max_length),
            dtype=np.float32,
            chunks=(1, min(128, max_length))
        )
        
        # Store additional metadata
        lengths_dataset = f.create_dataset(
            'lengths',
            shape=(valid_items,),
            dtype=np.int32
        )
        
        file_ids = f.create_dataset(
            'file_ids',
            shape=(valid_items,),
            dtype=h5py.special_dtype(vlen=str)
        )
        
        # Store audio parameters as attributes
        dataset.attrs['sample_rate'] = config['audio']['sample_rate']
        dataset.attrs['n_fft'] = config['audio']['n_fft']
        dataset.attrs['hop_length'] = config['audio']['hop_length']
        dataset.attrs['n_mels'] = config['audio']['n_mels']
        dataset.attrs['variable_length'] = True
        dataset.attrs['max_frames'] = max_length
        
        # Create phoneme datasets
        f.create_dataset('phoneme_starts', shape=(valid_items,), dtype=h5py.special_dtype(vlen=np.dtype('int32')))
        f.create_dataset('phoneme_ends', shape=(valid_items,), dtype=h5py.special_dtype(vlen=np.dtype('int32')))
        f.create_dataset('phoneme_texts', shape=(valid_items,), dtype=h5py.special_dtype(vlen=h5py.special_dtype(vlen=str)))
        f.create_dataset('phoneme_midi_pitches', shape=(valid_items,), dtype=h5py.special_dtype(vlen=np.dtype('float32')))
        
        idx = 0
        with tqdm(total=len(file_data), desc="Saving to H5", unit="file") as pbar:
            for file_id, file_info in file_data.items():
                if 'MEL_SPEC' in file_info and file_info['MEL_SPEC'] is not None:
                    mel_spec = file_info['MEL_SPEC']
                    mel_spec = normalize_mel_spectrogram(mel_spec)
                    
                    # Get the actual length (time frames)
                    actual_length = min(mel_spec.shape[1], max_length)
                    
                    # Store the actual length
                    lengths_dataset[idx] = actual_length
                    
                    # Pad or truncate to max_length
                    if mel_spec.shape[1] > max_length:
                        # Truncate if longer than max_length
                        mel_spec = mel_spec[:, :max_length]
                    elif mel_spec.shape[1] < max_length:
                        # Pad with zeros if shorter
                        padded = np.zeros((mel_bins, max_length), dtype=np.float32)
                        padded[:, :mel_spec.shape[1]] = mel_spec
                        mel_spec = padded
                    
                    # Store the mel spectrogram
                    dataset[idx] = mel_spec
                    file_ids[idx] = file_id
                    
                    # Store F0 data aligned with mel spectrogram
                    if 'F0_ALIGNED' in file_info and file_info['F0_ALIGNED'] is not None:
                        f0_aligned = file_info['F0_ALIGNED']
                        
                        # Ensure F0 matches the mel spectrogram time dimension
                        if len(f0_aligned) > max_length:
                            f0_aligned = f0_aligned[:max_length]
                        elif len(f0_aligned) < max_length:
                            f0_aligned = np.pad(f0_aligned, (0, max_length - len(f0_aligned)), 'constant')
                        
                        f0_dataset[idx] = f0_aligned
                    else:
                        # If no aligned F0 data, use zeros
                        f0_dataset[idx] = np.zeros(max_length, dtype=np.float32)
                    
                    # Save phoneme data including MIDI pitch if available
                    if 'PHONE_START' in file_info and 'MIDI_PITCH' in file_info:
                        f['phoneme_starts'][idx] = file_info['PHONE_START']
                        f['phoneme_ends'][idx] = file_info['PHONE_END']
                        f['phoneme_texts'][idx] = file_info['PHONE_TEXT']
                        f['phoneme_midi_pitches'][idx] = file_info['MIDI_PITCH']
                    
                    idx += 1
                
                pbar.update(1)
    
    logger.info(f"Saved {idx} mel spectrograms with aligned F0 data to {output_path}")

def collect_unique_phonemes(lab_files):
    unique_phonemes = set()
    
    with tqdm(total=len(lab_files), desc="Collecting phonemes", unit="file") as pbar:
        for file_path in lab_files:
            phonemes = parse_lab_file(file_path)
            for _, _, phone in phonemes:
                unique_phonemes.add(phone)
            pbar.update(1)
    
    phone_map = sorted(list(unique_phonemes))
    logger.info(f"Collected {len(phone_map)} unique phonemes")
    
    return phone_map

def main():
    parser = argparse.ArgumentParser(description='Process lab files and save data to H5 file')
    parser.add_argument('--config', type=str, default='config/model.yaml', help='Path to configuration file')
    parser.add_argument('--raw_dir', type=str, help='Raw directory path (overrides config)')
    parser.add_argument('--output', type=str, help='Path for the output H5 file (overrides config)')
    parser.add_argument('--min_phonemes', type=int, default=5, help='Minimum phonemes required per file')
    parser.add_argument('--data_key', type=str, default='mel_spectrograms', help='Key to use for data in the H5 file')
    parser.add_argument('--target_length', type=int, default=None, help='Target time frames (fixed length mode only)')
    parser.add_argument('--target_bins', type=int, default=None, help='Target mel bins')
    parser.add_argument('--variable_length', action='store_true', help='Enable variable length mel spectrograms')
    parser.add_argument('--max_audio_length', type=float, default=None, help='Maximum audio length in seconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional output')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.raw_dir:
        config['data']['raw_dir'] = args.raw_dir
    
    raw_dir = config['data']['raw_dir']
    
    output_path = args.output
    if output_path is None:
        bin_dir = config['data']['bin_dir']
        bin_file = config['data']['bin_file']
        output_path = os.path.join(bin_dir, bin_file)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Set variable length mode from args or config
    variable_length = args.variable_length
    if not variable_length and 'data' in config and 'variable_length' in config['data']:
        variable_length = config['data']['variable_length']
    
    # Set max audio length from args or config
    max_audio_length = args.max_audio_length
    if max_audio_length is None and 'audio' in config and 'max_audio_length' in config['audio']:
        max_audio_length = config['audio']['max_audio_length']
    
    if max_audio_length is not None:
        config['audio']['max_audio_length'] = max_audio_length
    else:
        config['audio']['max_audio_length'] = 10.0  # Default to 10 seconds
    
    # Configure target shape for fixed-length mode
    target_shape = None
    if args.target_length is not None and args.target_bins is not None:
        target_shape = (args.target_bins, args.target_length)
    
    lab_files = list_lab_files(raw_dir)
    
    phone_map = collect_unique_phonemes(lab_files)
    config['data']['phone_map'] = phone_map
    
    all_file_data = {}
    
    min_phoneme_count = args.min_phonemes
    skipped_files_count = 0
    processed_files_count = 0
    
    with tqdm(total=len(lab_files), desc="Processing files", unit="file") as pbar:
        for file_path in lab_files:
            phonemes = parse_lab_file(file_path)
            
            if len(phonemes) < min_phoneme_count:
                skipped_files_count += 1
                pbar.update(1)
                continue
            
            wav_file_path = find_wav_file(file_path, raw_dir)
            
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            file_id = base_filename
            
            mel_spec = None
            f0 = None
            f0_aligned = None
            
            if wav_file_path:
                if variable_length:
                    # Extract mel and F0 data with alignment
                    mel_spec, f0_aligned = extract_aligned_f0(wav_file_path, config)
                    
                    # If the above function fails, fall back to the original method
                    if mel_spec is None:
                        mel_spec = extract_mel_spectrogram_variable_length(wav_file_path, config)
                        
                        # IMPORTANT FIX: Get raw F0 in Hz, not MIDI
                        f0 = extract_f0(wav_file_path, config, convert_to_midi=False)
                else:
                    mel_spec = extract_mel_spectrogram(wav_file_path, config)
                    
                    # IMPORTANT FIX: Get raw F0 in Hz, not MIDI 
                    f0 = extract_f0(wav_file_path, config, convert_to_midi=False)
            
            phone_starts = np.array([p[0] for p in phonemes])
            phone_ends = np.array([p[1] for p in phonemes])
            phone_durations = phone_ends - phone_starts
            phone_texts = np.array([p[2] for p in phonemes], dtype=h5py.special_dtype(vlen=str))
            
            # Get the F0 data to use for MIDI pitch calculation
            f0_for_midi = f0_aligned if f0_aligned is not None else f0
            
            # Calculate MIDI pitch for each phoneme
            midi_pitches = []
            hop_length = config['audio']['hop_length']
            sample_rate = config['audio']['sample_rate']
            
            if f0_for_midi is not None and len(f0_for_midi) > 0:
                # Debug check for F0 format to detect potential issues
                if args.debug:
                    is_midi_format = is_midi_pitch(f0_for_midi)
                    logger.info(f"F0 data for {file_id} appears to be in {'MIDI' if is_midi_format else 'Hz'} format")
                    if is_midi_format:
                        logger.warning(f"File {file_id} has F0 data that looks like MIDI pitch already")
                
                # Process each phoneme
                for i, (start, end) in enumerate(zip(phone_starts, phone_ends)):
                    try:
                        # Calculate MIDI pitch using the improved function
                        midi_pitch = estimate_phoneme_midi_pitch(
                            f0_for_midi, start, end, hop_length, sample_rate
                        )
                        midi_pitches.append(midi_pitch)
                        
                        # Additional debug information
                        if args.debug and i < 5:  # Only for first few phonemes
                            # Convert time in samples to frame indices
                            start_frame = int(np.floor(start / hop_length))
                            end_frame = int(np.ceil(end / hop_length))
                            
                            # Ensure bounds are within the F0 array
                            start_frame = max(0, min(start_frame, len(f0_for_midi)-1))
                            end_frame = max(0, min(end_frame, len(f0_for_midi)))
                            
                            # Extract F0 values
                            phoneme_f0 = f0_for_midi[start_frame:end_frame]
                            voiced_f0 = phoneme_f0[phoneme_f0 > 0]
                            
                            # Log statistics
                            avg_f0 = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0
                            logger.info(f"  Phoneme {i} '{phone_texts[i]}': frames {start_frame}-{end_frame}, "
                                         f"avg F0: {avg_f0:.2f}, MIDI: {midi_pitch:.2f}")
                            
                    except Exception as e:
                        logger.error(f"Error calculating MIDI pitch for phoneme {i} in {file_id}: {e}")
                        midi_pitches.append(0.0)
            else:
                # If no F0 data, just use zeros
                midi_pitches = np.zeros(len(phone_starts), dtype=np.float32)
            
            # Ensure we have the same number of MIDI pitches as phonemes
            if len(midi_pitches) != len(phone_starts):
                logger.warning(f"MIDI pitch count ({len(midi_pitches)}) doesn't match phoneme count ({len(phone_starts)}) for {file_id}")
                # Fill with zeros if needed
                if len(midi_pitches) < len(phone_starts):
                    midi_pitches.extend([0.0] * (len(phone_starts) - len(midi_pitches)))
                else:
                    midi_pitches = midi_pitches[:len(phone_starts)]
                    
            midi_pitches = np.array(midi_pitches, dtype=np.float32)
            
            all_file_data[file_id] = {
                'PHONE_START': phone_starts,
                'PHONE_END': phone_ends,
                'PHONE_DURATION': phone_durations,
                'PHONE_TEXT': phone_texts,
                'MIDI_PITCH': midi_pitches,
                'FILE_NAME': np.array([file_path], dtype=h5py.special_dtype(vlen=str)),
                'MEL_SPEC': mel_spec,
                'F0': f0,
                'F0_ALIGNED': f0_aligned
            }
            
            processed_files_count += 1
            pbar.update(1)
    
    logger.info(f"Files processed: {processed_files_count}")
    logger.info(f"Files skipped: {skipped_files_count}")
    
    if all_file_data:
        if variable_length:
            save_to_h5_variable_length(output_path, all_file_data, phone_map, config, args.data_key)
        else:
            # Assuming the save_to_h5 function exists for fixed-length mode
            logger.warning("Fixed-length mode is not implemented in this script.")
            logger.info("Using variable length mode for saving.")
            save_to_h5_variable_length(output_path, all_file_data, phone_map, config, args.data_key)
    else:
        logger.warning("No files were processed. H5 file was not created.")

if __name__ == "__main__":
    main()