import os
import glob
import h5py
import numpy as np
import math
from tqdm import tqdm
import argparse
import logging
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image

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

def time_to_frame(time_htk, sample_rate, hop_length):
    """
    Convert HTK-style time units (100ns) to mel spectrogram frame index
    
    Args:
        time_htk: Time in HTK units (100ns)
        sample_rate: Audio sample rate
        hop_length: Hop length used in STFT
        
    Returns:
        Frame index
    """
    # Convert HTK time units (100ns) to seconds
    time_seconds = time_htk * 1e-7
    
    # Convert seconds to samples
    time_samples = time_seconds * sample_rate
    
    # Convert samples to frames
    frame_index = time_samples / hop_length
    
    return int(frame_index)

def frequency_to_midi_pitch(frequency):
    """
    Convert frequency (Hz) to MIDI pitch number
    
    Args:
        frequency: Frequency in Hz
        
    Returns:
        MIDI pitch number (0-127)
    """
    if frequency <= 0:
        return 0  # MIDI note 0 for silence or undefined
    
    # Formula to convert frequency to MIDI pitch
    midi_pitch = 69 + 12 * math.log2(frequency / 440.0)
    
    # Constrain to valid MIDI range (0-127)
    midi_pitch = max(0, min(127, int(round(midi_pitch))))
    
    return midi_pitch

def calculate_phoneme_pitches(frame_starts, frame_ends, f0_values):
    """
    Calculate average MIDI pitch for each phoneme based on F0 values
    
    Args:
        frame_starts: Array of phoneme start frames
        frame_ends: Array of phoneme end frames
        f0_values: F0 values aligned with frames
        
    Returns:
        Array of MIDI pitch values for each phoneme
    """
    midi_pitches = []
    
    for start, end in zip(frame_starts, frame_ends):
        # Get F0 values for this phoneme duration
        phoneme_f0 = f0_values[start:end]
        
        # Filter out zero values (unvoiced)
        voiced_f0 = phoneme_f0[phoneme_f0 > 0]
        
        if len(voiced_f0) > 0:
            # Calculate average F0 for voiced segments
            avg_f0 = np.mean(voiced_f0)
            # Convert to MIDI pitch
            midi_pitch = frequency_to_midi_pitch(avg_f0)
        else:
            # Unvoiced phoneme
            midi_pitch = 0
        
        midi_pitches.append(midi_pitch)
    
    return np.array(midi_pitches, dtype=np.int16)

def extract_f0_aligned(wav_path, config, mel_spec_length=None):
    """
    Extract F0 values aligned with mel spectrogram time frames
    
    Args:
        wav_path: Path to the wav file
        config: Configuration dictionary
        mel_spec_length: Length of the mel spectrogram in time frames
        
    Returns:
        F0 values aligned with mel spectrogram time frames
    """
    try:
        import numpy as np
        import librosa
        
        y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
        
        # Check if audio exceeds maximum length
        max_audio_length = config['audio'].get('max_audio_length', 10.0)
        max_samples = int(max_audio_length * sr)
        
        if len(y) > max_samples:
            print(f"Warning: Audio file {wav_path} exceeds maximum length of {max_audio_length}s. Truncating.")
            y = y[:max_samples]
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=config['audio'].get('f0_min', 50),
            fmax=config['audio'].get('f0_max', 600),
            sr=sr,
            hop_length=config['audio']['hop_length']
        )
        
        f0 = np.nan_to_num(f0)
        
        # If mel_spec_length is provided, align F0 length to match
        if mel_spec_length is not None:
            if len(f0) > mel_spec_length:
                # Truncate F0 if longer
                logger.info(f"Truncating F0 from {len(f0)} to {mel_spec_length} frames to match mel spectrogram")
                f0 = f0[:mel_spec_length]
            elif len(f0) < mel_spec_length:
                # Pad F0 with zeros if shorter
                logger.info(f"Padding F0 from {len(f0)} to {mel_spec_length} frames to match mel spectrogram")
                padded_f0 = np.zeros(mel_spec_length)
                padded_f0[:len(f0)] = f0
                f0 = padded_f0
        
        return f0
        
    except Exception as e:
        logger.error(f"Error extracting F0 from {wav_path}: {e}")
        return None
    
def align_phonemes_to_frames(phonemes, mel_spec_length, sample_rate, hop_length):
    """
    Align phoneme boundaries to frame indices in mel spectrogram
    
    Args:
        phonemes: List of (start_time, end_time, phoneme) tuples from lab file
        mel_spec_length: The number of frames in the mel spectrogram (time dimension)
        sample_rate: Audio sample rate
        hop_length: Hop length used in STFT
        
    Returns:
        Tuple of (frame_starts, frame_ends) arrays
    """
    frame_starts = []
    frame_ends = []
    phone_texts = []
    
    for start_time, end_time, phoneme in phonemes:
        # Convert HTK time units to frames
        start_frame = time_to_frame(start_time, sample_rate, hop_length)
        end_frame = time_to_frame(end_time, sample_rate, hop_length)
        
        # Ensure frames are within mel spectrogram bounds
        start_frame = max(0, min(start_frame, mel_spec_length - 1))
        end_frame = max(start_frame + 1, min(end_frame, mel_spec_length))
        
        frame_starts.append(start_frame)
        frame_ends.append(end_frame)
        phone_texts.append(phoneme)
    
    return np.array(frame_starts), np.array(frame_ends), phone_texts

def verify_phoneme_alignment(mel_spec, frame_starts, frame_ends, phone_texts, file_id, 
                           output_dir=None, visualize=False, f0=None):
    """
    Verify the quality of phoneme-to-frame alignment and optionally check F0 alignment
    
    Args:
        mel_spec: Mel spectrogram (freq_bins, time_frames)
        frame_starts: Array of phoneme start frames
        frame_ends: Array of phoneme end frames
        phone_texts: List of phoneme labels
        file_id: File identifier for logging
        output_dir: Directory to save visualization plots (if visualize=True)
        visualize: Whether to generate visualization plots
        f0: F0 values aligned with mel spectrogram (optional)
        
    Returns:
        Dictionary with alignment quality metrics
    """
    if len(frame_starts) == 0:
        logger.warning(f"No phonemes to verify in file {file_id}")
        return {"status": "no_phonemes"}
    
    # Check for ordering issues
    ordered = True
    for i in range(len(frame_starts) - 1):
        if frame_ends[i] > frame_starts[i+1]:
            ordered = False
            logger.warning(f"Phoneme ordering issue in {file_id}: {phone_texts[i]} and {phone_texts[i+1]} overlap")
    
    # Check for coverage
    total_frames = mel_spec.shape[1]
    covered_frames = 0
    for start, end in zip(frame_starts, frame_ends):
        covered_frames += (end - start)
    
    coverage_percent = (covered_frames / total_frames) * 100
    
    # Check for gaps
    gaps = []
    for i in range(len(frame_starts) - 1):
        gap = frame_starts[i+1] - frame_ends[i]
        if gap > 0:
            gaps.append((gap, i))
    
    # Check for tiny segments
    tiny_segments = []
    for i, (start, end) in enumerate(zip(frame_starts, frame_ends)):
        if (end - start) <= 1:
            tiny_segments.append((i, phone_texts[i]))
    
    # Check F0 alignment if provided
    f0_aligned = False
    if f0 is not None:
        f0_aligned = len(f0) == total_frames
    
    # Prepare results
    results = {
        "file_id": file_id,
        "total_frames": total_frames,
        "total_phonemes": len(frame_starts),
        "coverage_percent": coverage_percent,
        "ordered": ordered,
        "gaps": len(gaps),
        "tiny_segments": len(tiny_segments),
        "f0_aligned": f0_aligned,
        "status": "verified"
    }
    
    # Log a summary
    logger.info(f"Alignment verification for {file_id}: " +
                f"{len(frame_starts)} phonemes, " +
                f"{coverage_percent:.1f}% coverage, " +
                f"ordered: {ordered}, " +
                f"gaps: {len(gaps)}, " +
                f"tiny segments: {len(tiny_segments)}" +
                (f", F0 aligned: {f0_aligned}" if f0 is not None else ""))
    
    # Create visualization if requested
    if visualize and output_dir is not None:
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a visualization of the alignment
            if f0 is not None:
                # Create plot with 3 subplots: mel, f0, and phoneme alignment
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), 
                                                   gridspec_kw={'height_ratios': [3, 1, 1]})
            else:
                # Create plot with 2 subplots: mel and phoneme alignment
                fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(12, 8), 
                                              gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot the mel spectrogram
            im = ax1.imshow(mel_spec, aspect='auto', origin='lower', cmap='viridis')
            ax1.set_title(f"Mel Spectrogram and Phoneme Alignment: {file_id}")
            ax1.set_ylabel("Mel Bins")
            fig.colorbar(im, ax=ax1)
            
            # Plot F0 if provided
            if f0 is not None:
                time_frames = np.arange(len(f0))
                ax2.plot(time_frames, f0, 'r-')
                ax2.set_ylabel("F0 (Hz)")
                ax2.set_xlim(0, len(f0))
                # Draw vertical lines at phoneme boundaries
                for start, end in zip(frame_starts, frame_ends):
                    ax2.axvline(x=start, color='blue', linestyle='--', alpha=0.3)
                    ax2.axvline(x=end, color='green', linestyle='--', alpha=0.3)
            
            # Create a colormap for phoneme visualization
            n_phonemes = len(frame_starts)
            colors = plt.cm.jet(np.linspace(0, 1, n_phonemes))
            
            # Create a phoneme segment visualization
            phoneme_viz = np.zeros((50, total_frames))
            for i, (start, end, phoneme) in enumerate(zip(frame_starts, frame_ends, phone_texts)):
                phoneme_viz[:, start:end] = i + 1
            
            # Create a custom colormap with a transparent background for unaligned frames
            cmap = plt.cm.jet
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
            cmap.set_under('white', alpha=0.5)  # Color for values < vmin
            
            # Plot the phoneme segmentation
            im2 = ax3.imshow(phoneme_viz, aspect='auto', origin='lower', cmap=cmap, 
                            vmin=0.5, vmax=n_phonemes+0.5)
            ax3.set_yticks([25])
            ax3.set_yticklabels(["Phonemes"])
            ax3.set_xlabel("Time Frames")
            
            # Add phoneme text labels
            prev_end = -100  # To avoid overlapping text
            for i, (start, end, phoneme) in enumerate(zip(frame_starts, frame_ends, phone_texts)):
                mid = (start + end) // 2
                if mid - prev_end > 20:  # Only add text if there's enough space
                    ax3.text(mid, 25, phoneme, ha='center', va='center', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                    prev_end = mid
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{file_id}_alignment.png"), dpi=150)
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating alignment visualization for {file_id}: {e}")
    
    return results

def save_to_h5_variable_length(output_path, file_data, phone_map, config, data_key='mel_spectrograms'):
    """Save mel spectrograms to H5 file with variable length support"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mel_bins = config['model'].get('mel_bins', 80)
    sample_rate = config['audio'].get('sample_rate', 22050)
    hop_length = config['audio'].get('hop_length', 256)
    
    # Get alignment verification settings
    verify_alignment = config.get('preprocessing', {}).get('verify_alignment', False)
    visualize_alignment = config.get('preprocessing', {}).get('visualize_alignment', False)
    alignment_vis_dir = config.get('preprocessing', {}).get('alignment_vis_dir', 'alignment_viz')
    
    if visualize_alignment:
        os.makedirs(alignment_vis_dir, exist_ok=True)
    
    # Calculate max possible time frames for the maximum audio length
    max_audio_length = config['audio'].get('max_audio_length', 10.0)  # Default 10 seconds
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
    
    # Prepare for alignment statistics
    if verify_alignment:
        alignment_stats = {
            'total_files': 0,
            'coverage': [],
            'ordered_count': 0,
            'unordered_count': 0,
            'files_with_gaps': 0,
            'files_with_tiny_segments': 0,
            'f0_alignment': []  # Track F0 alignment success rate
        }
    
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
        
        # New datasets for frame-aligned phoneme boundaries
        phone_frame_starts = f.create_dataset(
            'phone_frame_starts',
            shape=(valid_items,),
            dtype=h5py.vlen_dtype(np.int32)
        )
        
        phone_frame_ends = f.create_dataset(
            'phone_frame_ends',
            shape=(valid_items,),
            dtype=h5py.vlen_dtype(np.int32)
        )
        
        phone_texts = f.create_dataset(
            'phone_texts',
            shape=(valid_items,),
            dtype=h5py.vlen_dtype(h5py.special_dtype(vlen=str))
        )
        
        # Add new dataset for MIDI pitch values
        midi_pitches = f.create_dataset(
            'MIDI_PITCH',
            shape=(valid_items,),
            dtype=h5py.vlen_dtype(np.int16)
        )
        
        # Dataset for F0 values
        f0_dataset = f.create_dataset(
            'f0_values',
            shape=(valid_items, max_length),
            dtype=np.float32,
            chunks=(1, min(128, max_length))
        )
        
        # Dataset for alignment verification results if enabled
        if verify_alignment:
            alignment_quality = f.create_dataset(
                'alignment_quality',
                shape=(valid_items,),
                dtype=np.float32
            )
        
        # Store audio parameters as attributes
        dataset.attrs['sample_rate'] = config['audio']['sample_rate']
        dataset.attrs['n_fft'] = config['audio']['n_fft']
        dataset.attrs['hop_length'] = config['audio']['hop_length']
        dataset.attrs['n_mels'] = config['audio']['n_mels']
        dataset.attrs['variable_length'] = True
        dataset.attrs['max_frames'] = max_length
        
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
                    
                    # Convert phoneme times to frame indices
                    phoneme_times = list(zip(file_info['PHONE_START'], 
                                            file_info['PHONE_END'], 
                                            file_info['PHONE_TEXT']))
                    
                    frame_starts, frame_ends, phoneme_label_list = align_phonemes_to_frames(
                        phoneme_times,
                        actual_length,
                        sample_rate,
                        hop_length
                    )
                    
                    # Store the frame indices and phoneme labels
                    phone_frame_starts[idx] = frame_starts
                    phone_frame_ends[idx] = frame_ends
                    phone_texts[idx] = np.array(phoneme_label_list)
                    
                    # Prepare F0 values
                    if 'F0' in file_info and file_info['F0'] is not None:
                        f0 = file_info['F0']
                        
                        # Ensure F0 aligns with mel spectrogram
                        if len(f0) > actual_length:
                            f0 = f0[:actual_length]
                        elif len(f0) < actual_length:
                            padded_f0 = np.zeros(actual_length)
                            padded_f0[:len(f0)] = f0
                            f0 = padded_f0
                        
                        # Calculate MIDI pitch for each phoneme based on F0
                        midi_pitch_values = calculate_phoneme_pitches(frame_starts, frame_ends, f0)
                        
                        # Store MIDI pitch values
                        midi_pitches[idx] = midi_pitch_values
                        
                        # Pad to max_length
                        if len(f0) < max_length:
                            padded = np.zeros(max_length, dtype=np.float32)
                            padded[:len(f0)] = f0
                            f0 = padded
                    else:
                        # Create empty F0 values
                        f0 = np.zeros(max_length, dtype=np.float32)
                        # Create empty MIDI pitch values (all zeros)
                        midi_pitches[idx] = np.zeros(len(frame_starts), dtype=np.int16)
                    
                    # Store F0 values
                    f0_dataset[idx] = f0
                    
                    # Verify alignment if enabled
                    if verify_alignment:
                        alignment_results = verify_phoneme_alignment(
                            mel_spec, 
                            frame_starts, 
                            frame_ends, 
                            phoneme_label_list,
                            file_id,
                            alignment_vis_dir if visualize_alignment else None,
                            visualize=visualize_alignment,
                            f0=file_info.get('F0')  # Pass F0 for visualization
                        )
                        
                        # Store alignment quality metric (coverage percentage)
                        alignment_quality[idx] = alignment_results['coverage_percent']
                        
                        # Update alignment statistics
                        if alignment_results['status'] == 'verified':
                            alignment_stats['total_files'] += 1
                            alignment_stats['coverage'].append(alignment_results['coverage_percent'])
                            alignment_stats['ordered_count'] += (1 if alignment_results['ordered'] else 0)
                            alignment_stats['unordered_count'] += (0 if alignment_results['ordered'] else 1)
                            alignment_stats['files_with_gaps'] += (1 if alignment_results['gaps'] > 0 else 0)
                            alignment_stats['files_with_tiny_segments'] += (1 if alignment_results['tiny_segments'] > 0 else 0)
                            if 'f0_aligned' in alignment_results:
                                alignment_stats['f0_alignment'].append(alignment_results['f0_aligned'])
                    
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
                    idx += 1
                
                pbar.update(1)
    
    logger.info(f"Saved {idx} mel spectrograms to {output_path} with variable length support")
    logger.info(f"Added frame-aligned phoneme boundaries, F0 values, and MIDI pitch values")
    
    # Log alignment statistics if verification was enabled
    if verify_alignment and alignment_stats['total_files'] > 0:
        mean_coverage = sum(alignment_stats['coverage']) / len(alignment_stats['coverage'])
        logger.info(f"Alignment statistics:")
        logger.info(f"  - Files processed: {alignment_stats['total_files']}")
        logger.info(f"  - Average coverage: {mean_coverage:.2f}%")
        logger.info(f"  - Files with correctly ordered phonemes: {alignment_stats['ordered_count']} ({alignment_stats['ordered_count']/alignment_stats['total_files']*100:.1f}%)")
        logger.info(f"  - Files with gaps between phonemes: {alignment_stats['files_with_gaps']} ({alignment_stats['files_with_gaps']/alignment_stats['total_files']*100:.1f}%)")
        logger.info(f"  - Files with tiny phoneme segments: {alignment_stats['files_with_tiny_segments']} ({alignment_stats['files_with_tiny_segments']/alignment_stats['total_files']*100:.1f}%)")
        
        if alignment_stats['f0_alignment']:
            f0_success_rate = sum(alignment_stats['f0_alignment']) / len(alignment_stats['f0_alignment']) * 100
            logger.info(f"  - F0 alignment success rate: {f0_success_rate:.1f}%")
        
        if visualize_alignment:
            logger.info(f"Alignment visualizations saved to: {alignment_vis_dir}")

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
    parser.add_argument('--verify_alignment', action='store_true', help='Verify phoneme-to-frame alignment')
    parser.add_argument('--visualize_alignment', action='store_true', help='Generate alignment visualizations')
    parser.add_argument('--alignment_vis_dir', type=str, default='alignment_viz', help='Directory for alignment visualizations')
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
    
    # Configure alignment verification settings
    if 'preprocessing' not in config:
        config['preprocessing'] = {}
    
    config['preprocessing']['verify_alignment'] = args.verify_alignment
    config['preprocessing']['visualize_alignment'] = args.visualize_alignment
    config['preprocessing']['alignment_vis_dir'] = args.alignment_vis_dir
    
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
            if wav_file_path:
                if variable_length:
                    mel_spec = extract_mel_spectrogram_variable_length(wav_file_path, config)
                else:
                    mel_spec = extract_mel_spectrogram(wav_file_path, config)
                
                # Extract F0 aligned with mel spectrogram time dimension
                if mel_spec is not None:
                    f0 = extract_f0_aligned(wav_file_path, config, mel_spec.shape[1])
                else:
                    f0 = extract_f0(wav_file_path, config)
            
            phone_starts = np.array([p[0] for p in phonemes])
            phone_ends = np.array([p[1] for p in phonemes])
            phone_durations = phone_ends - phone_starts
            phone_texts = np.array([p[2] for p in phonemes], dtype=h5py.special_dtype(vlen=str))
            
            all_file_data[file_id] = {
                'PHONE_START': phone_starts,
                'PHONE_END': phone_ends,
                'PHONE_DURATION': phone_durations,
                'PHONE_TEXT': phone_texts,
                'FILE_NAME': np.array([file_path], dtype=h5py.special_dtype(vlen=str)),
                'MEL_SPEC': mel_spec,
                'F0': f0
            }
            
            processed_files_count += 1
            pbar.update(1)
    
    logger.info(f"Files processed: {processed_files_count}")
    logger.info(f"Files skipped: {skipped_files_count}")
    
    if all_file_data:
        if variable_length:
            save_to_h5_variable_length(output_path, all_file_data, phone_map, config, args.data_key)
        else:
            # Original save_to_h5 function (not modified for this example)
            logger.warning("Fixed-length mode does not support frame-aligned phoneme boundaries yet.")
            save_to_h5(output_path, all_file_data, phone_map, config, args.data_key, target_shape)
    else:
        logger.warning("No files were processed. H5 file was not created.")

if __name__ == "__main__":
    main()