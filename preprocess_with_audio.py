import os
import glob
import h5py
import numpy as np
import math
from tqdm import tqdm
import argparse
import logging
import soundfile as sf
import concurrent.futures
import functools

from utils.utils import (
    load_config, 
    extract_mel_spectrogram,
    extract_mel_spectrogram_variable_length,
    extract_f0, 
    normalize_mel_spectrogram, 
    pad_or_truncate_mel
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('preprocess')

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

def resample_audio_fast(y, orig_sr, target_sr):
    """Faster audio resampling using pydub or scipy if available, falling back to librosa"""
    try:
        # Try using scipy for resampling
        from scipy import signal
        
        # Calculate number of samples needed for target sample rate
        target_samples = int(y.shape[0] * target_sr / orig_sr)
        
        # Use scipy's resample which is generally faster than librosa
        resampled = signal.resample(y, target_samples)
        return resampled
    except ImportError:
        try:
            # Try using pydub for resampling
            from pydub import AudioSegment
            import io
            
            # Convert numpy array to pydub AudioSegment
            y_int16 = (y * 32767).astype(np.int16)
            byte_io = io.BytesIO()
            sf.write(byte_io, y_int16, orig_sr, format='wav')
            byte_io.seek(0)
            audio_segment = AudioSegment.from_file(byte_io, format="wav")
            
            # Resample using pydub (faster than librosa)
            audio_segment = audio_segment.set_frame_rate(target_sr)
            
            # Convert back to numpy array
            samples = np.array(audio_segment.get_array_of_samples())
            return samples.astype(np.float32) / 32767.0
        except ImportError:
            # Fall back to librosa if neither scipy nor pydub is available
            import librosa
            return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

def extract_audio_waveform(wav_file_path, config, dtype=np.int16, use_fast_resample=True):
    """Extract audio waveform from WAV file with optimized processing"""
    try:
        # Load audio with soundfile (faster than librosa.load for WAV files)
        y, sr = sf.read(wav_file_path, always_2d=False)
        
        # Check if we need to resample
        target_sr = config['audio']['sample_rate']
        if sr != target_sr:
            if use_fast_resample:
                y = resample_audio_fast(y, sr, target_sr)
            else:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Check if audio exceeds maximum length
        max_audio_length = config['audio'].get('max_audio_length', 10.0)
        max_samples = int(max_audio_length * target_sr)
        
        if len(y) > max_samples:
            y = y[:max_samples]
        
        # Convert to mono if needed - use faster approach with reshape+mean
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = np.mean(y, axis=1)
        
        # Ensure output is 1D
        y = np.squeeze(y)
        
        # Convert to target dtype
        if dtype == np.int16:
            if y.dtype == np.float32 or y.dtype == np.float64:
                # Faster approach directly using numpy operations
                y = np.clip(y * 32767, -32768, 32767).astype(np.int16)
            else:
                y = y.astype(np.int16)
        else:
            y = y.astype(dtype)
        
        return y
    
    except Exception as e:
        logger.error(f"Error extracting audio from {wav_file_path}: {e}")
        return None

def process_file_parallel(file_path, config, raw_dir, min_phoneme_count, variable_length):
    """Process a single file - function suitable for parallel processing"""
    phonemes = parse_lab_file(file_path)
    
    if len(phonemes) < min_phoneme_count:
        return None
    
    wav_file_path = find_wav_file(file_path, raw_dir)
    
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    file_id = base_filename
    
    mel_spec = None
    audio = None
    f0 = None
    
    if wav_file_path:
        if variable_length:
            mel_spec = extract_mel_spectrogram_variable_length(wav_file_path, config)
        else:
            mel_spec = extract_mel_spectrogram(wav_file_path, config)
        
        # Extract audio waveform as int16 with fast resampling
        audio = extract_audio_waveform(wav_file_path, config, dtype=np.int16, use_fast_resample=True)
        
        # Extract F0 if needed
        f0 = extract_f0(wav_file_path, config)
    
    phone_starts = np.array([p[0] for p in phonemes])
    phone_ends = np.array([p[1] for p in phonemes])
    phone_durations = phone_ends - phone_starts
    phone_texts = np.array([p[2] for p in phonemes], dtype=h5py.special_dtype(vlen=str))
    
    return {
        'file_id': file_id,
        'PHONE_START': phone_starts,
        'PHONE_END': phone_ends,
        'PHONE_DURATION': phone_durations,
        'PHONE_TEXT': phone_texts,
        'FILE_NAME': np.array([file_path], dtype=h5py.special_dtype(vlen=str)),
        'MEL_SPEC': mel_spec,
        'AUDIO': audio,
        'F0': f0
    }

def save_to_h5_with_audio(output_path, file_data, phone_map, config, 
                         data_key='mel_spectrograms', audio_key='waveforms'):
    """Save mel spectrograms and audio to H5 file with variable length support"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mel_bins = config['model'].get('mel_bins', 80)
    
    # Calculate max possible time frames for the maximum audio length
    max_audio_length = config['audio'].get('max_audio_length', 10.0)
    sample_rate = config['audio']['sample_rate']
    hop_length = config['audio']['hop_length']
    max_time_frames = math.ceil(max_audio_length * sample_rate / hop_length)
    
    # Count valid items with both mel and audio
    valid_items = 0
    max_mel_length = 0
    max_audio_length_samples = 0
    
    for file_info in file_data.values():
        if ('MEL_SPEC' in file_info and file_info['MEL_SPEC'] is not None and
            'AUDIO' in file_info and file_info['AUDIO'] is not None):
            valid_items += 1
            curr_mel_length = file_info['MEL_SPEC'].shape[1]  # Time dimension
            max_mel_length = max(max_mel_length, curr_mel_length)
            
            curr_audio_length = len(file_info['AUDIO'])
            max_audio_length_samples = max(max_audio_length_samples, curr_audio_length)
    
    # Cap the maximum length to max_time_frames
    max_mel_length = min(max_mel_length, max_time_frames)
    logger.info(f"Maximum mel time frames: {max_mel_length} (equivalent to {max_mel_length * hop_length / sample_rate:.2f} seconds)")
    logger.info(f"Maximum audio length: {max_audio_length_samples} samples (equivalent to {max_audio_length_samples / sample_rate:.2f} seconds)")
    
    # Define compression settings for better performance
    compression_opts = {
        'compression': 'gzip',
        'compression_opts': 1  # Low compression level for better write speed
    }
    
    with h5py.File(output_path, 'w') as f:
        # Store phone map
        phone_map_array = np.array(phone_map, dtype=h5py.special_dtype(vlen=str))
        f.create_dataset('phone_map', data=phone_map_array)
        
        # Create datasets with optimized settings
        mel_dataset = f.create_dataset(
            data_key,
            shape=(valid_items, mel_bins, max_mel_length),
            dtype=np.float32,
            chunks=(1, mel_bins, min(128, max_mel_length)),
            **compression_opts
        )
        
        audio_dataset = f.create_dataset(
            audio_key,
            shape=(valid_items, max_audio_length_samples),
            dtype=np.int16,
            chunks=(1, min(8192, max_audio_length_samples)),
            **compression_opts
        )
        
        # Store additional metadata
        lengths_dataset = f.create_dataset(
            'lengths',
            shape=(valid_items, 2),  # Store both mel and audio lengths
            dtype=np.int32
        )
        
        file_ids = f.create_dataset(
            'file_ids',
            shape=(valid_items,),
            dtype=h5py.special_dtype(vlen=str)
        )
        
        # Store audio parameters as attributes
        mel_dataset.attrs['sample_rate'] = config['audio']['sample_rate']
        mel_dataset.attrs['n_fft'] = config['audio']['n_fft']
        mel_dataset.attrs['hop_length'] = config['audio']['hop_length']
        mel_dataset.attrs['n_mels'] = config['audio']['n_mels']
        mel_dataset.attrs['variable_length'] = True
        mel_dataset.attrs['max_frames'] = max_mel_length
        
        audio_dataset.attrs['sample_rate'] = config['audio']['sample_rate']
        audio_dataset.attrs['max_samples'] = max_audio_length_samples
        
        # Process in larger chunks for more efficient HDF5 writing
        chunk_size = 100  # Adjust based on your memory constraints
        file_ids_list = list(file_data.keys())
        num_chunks = (len(file_ids_list) + chunk_size - 1) // chunk_size
        
        idx = 0
        with tqdm(total=len(file_data), desc="Saving to H5", unit="file") as pbar:
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, len(file_ids_list))
                chunk_file_ids = file_ids_list[start_idx:end_idx]
                
                # Process files in this chunk
                for file_id in chunk_file_ids:
                    file_info = file_data[file_id]
                    
                    if ('MEL_SPEC' in file_info and file_info['MEL_SPEC'] is not None and
                        'AUDIO' in file_info and file_info['AUDIO'] is not None):
                        # Process mel spectrogram
                        mel_spec = file_info['MEL_SPEC']
                        mel_spec = normalize_mel_spectrogram(mel_spec)
                        
                        # Get the actual length (time frames)
                        actual_mel_length = min(mel_spec.shape[1], max_mel_length)
                        
                        # Process audio waveform
                        audio = file_info['AUDIO']
                        # Ensure audio is 1D
                        audio = np.squeeze(audio)
                        actual_audio_length = min(len(audio), max_audio_length_samples)
                        
                        # Store the actual lengths
                        lengths_dataset[idx, 0] = actual_mel_length  # Mel length
                        lengths_dataset[idx, 1] = actual_audio_length  # Audio length
                        
                        # Pad or truncate mel spectrogram and audio
                        if mel_spec.shape[1] > max_mel_length:
                            mel_spec = mel_spec[:, :max_mel_length]
                        elif mel_spec.shape[1] < max_mel_length:
                            padded_mel = np.zeros((mel_bins, max_mel_length), dtype=np.float32)
                            padded_mel[:, :mel_spec.shape[1]] = mel_spec
                            mel_spec = padded_mel
                        
                        if len(audio) > max_audio_length_samples:
                            audio = audio[:max_audio_length_samples]
                        elif len(audio) < max_audio_length_samples:
                            padded_audio = np.zeros(max_audio_length_samples, dtype=np.int16)
                            padded_audio[:len(audio)] = audio
                            audio = padded_audio
                        
                        # Store the data
                        mel_dataset[idx] = mel_spec
                        audio_dataset[idx] = audio
                        file_ids[idx] = file_id
                        idx += 1
                    
                    pbar.update(1)
    
    logger.info(f"Saved {idx} mel spectrograms and audio waveforms to {output_path}")

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
    parser = argparse.ArgumentParser(description='Process lab files and save data to H5 file with audio (optimized)')
    parser.add_argument('--config', type=str, default='config/model_with_vocoder.yaml', help='Path to configuration file')
    parser.add_argument('--raw_dir', type=str, help='Raw directory path (overrides config)')
    parser.add_argument('--output', type=str, help='Path for the output H5 file (overrides config)')
    parser.add_argument('--min_phonemes', type=int, default=5, help='Minimum phonemes required per file')
    parser.add_argument('--data_key', type=str, default='mel_spectrograms', help='Key to use for mel data in the H5 file')
    parser.add_argument('--audio_key', type=str, default='waveforms', help='Key to use for audio data in the H5 file')
    parser.add_argument('--variable_length', action='store_true', help='Enable variable length processing')
    parser.add_argument('--max_audio_length', type=float, default=None, help='Maximum audio length in seconds')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--chunk_size', type=int, default=20, help='Number of files to process in each parallel batch')
    parser.add_argument('--no_parallel', action='store_true', help='Disable parallel processing')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.raw_dir:
        config['data']['raw_dir'] = args.raw_dir
    
    raw_dir = config['data']['raw_dir']
    
    output_path = args.output
    if output_path is None:
        bin_dir = config['data']['bin_dir']
        bin_file = config['data'].get('bin_file', 'mel_spectrograms_with_audio.h5')
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
    
    lab_files = list_lab_files(raw_dir)
    
    phone_map = collect_unique_phonemes(lab_files)
    config['data']['phone_map'] = phone_map
    
    all_file_data = {}
    skipped_files_count = 0
    processed_files_count = 0
    audio_missing_count = 0
    
    # Process files (either in parallel or sequentially)
    if args.no_parallel:
        logger.info("Parallel processing disabled, processing files sequentially")
        
        with tqdm(total=len(lab_files), desc="Processing files", unit="file") as pbar:
            for file_path in lab_files:
                result = process_file_parallel(
                    file_path, 
                    config, 
                    raw_dir, 
                    args.min_phonemes,
                    variable_length
                )
                
                if result is not None:
                    file_id = result.pop('file_id')
                    all_file_data[file_id] = result
                    processed_files_count += 1
                    if result['AUDIO'] is None:
                        audio_missing_count += 1
                else:
                    skipped_files_count += 1
                
                pbar.update(1)
    else:
        # Set up parallelization
        num_workers = args.num_workers
        if num_workers is None:
            # Use CPU count - 1 to avoid maxing out the CPU
            import multiprocessing
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        
        logger.info(f"Processing files using {num_workers} workers in parallel")
        
        # Partial function with fixed arguments
        process_func = functools.partial(
            process_file_parallel, 
            config=config, 
            raw_dir=raw_dir, 
            min_phoneme_count=args.min_phonemes,
            variable_length=variable_length
        )
        
        # Process in batches for better progress tracking
        chunk_size = args.chunk_size
        num_chunks = (len(lab_files) + chunk_size - 1) // chunk_size
        
        with tqdm(total=len(lab_files), desc="Processing files", unit="file") as progress_bar:
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, len(lab_files))
                chunk_files = lab_files[start_idx:end_idx]
                
                # Process this batch in parallel
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                    results = list(executor.map(process_func, chunk_files))
                    
                    # Process results
                    for result in results:
                        if result is not None:
                            file_id = result.pop('file_id')
                            all_file_data[file_id] = result
                            processed_files_count += 1
                            if result['AUDIO'] is None:
                                audio_missing_count += 1
                        else:
                            skipped_files_count += 1
                        
                        progress_bar.update(1)
    
    logger.info(f"Files processed: {processed_files_count}")
    logger.info(f"Files skipped: {skipped_files_count}")
    logger.info(f"Files with missing audio: {audio_missing_count}")
    
    if all_file_data:
        save_to_h5_with_audio(
            output_path, 
            all_file_data, 
            phone_map, 
            config,
            args.data_key,
            args.audio_key
        )
    else:
        logger.warning("No files were processed. H5 file was not created.")

if __name__ == "__main__":
    main()