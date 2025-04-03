import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import librosa
import soundfile as sf

from utils.utils import load_config
from models.conditional_multi_band_unet import ConditionalMultiBandUNet
from data.conditional_dataset import H5FileManager

def load_phone_map(h5_path):
    """Load phoneme map from H5 file"""
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'phone_map' in f:
                return list(f['phone_map'][:])
    except Exception as e:
        print(f"Error loading phone map: {e}")
    
    return []

def prep_conditioning_from_file(h5_path, idx, max_frames=None):
    """
    Prepare conditioning information from H5 file for a specific sample
    
    Args:
        h5_path: Path to H5 file
        idx: Index of sample to use
        max_frames: Maximum number of frames to use
    
    Returns:
        Dictionary of conditioning tensors
    """
    try:
        with h5py.File(h5_path, 'r') as f:
            # Load phoneme information
            has_phone_data = ('phone_frame_starts' in f and 
                             'phone_frame_ends' in f and 
                             'phone_texts' in f)
            
            # Load phoneme map
            phone_map = list(f['phone_map'][:]) if 'phone_map' in f else []
            
            # Get actual length
            if 'lengths' in f:
                time_frames = f['lengths'][idx]
            else:
                # Try to get from mel spectrograms
                if 'mel_spectrograms' in f:
                    time_frames = f['mel_spectrograms'].shape[2]
                else:
                    time_frames = 1000  # Default if not available
            
            # Limit to max_frames if specified
            if max_frames and time_frames > max_frames:
                time_frames = max_frames
            
            # Initialize conditioning dictionary
            conditioning = {}
            
            # Get phoneme information
            if has_phone_data:
                phone_starts = f['phone_frame_starts'][idx]
                phone_ends = f['phone_frame_ends'][idx]
                phone_texts = f['phone_texts'][idx]
                
                # Convert phoneme text to phoneme IDs
                phone_ids = []
                for phone in phone_texts:
                    if phone in phone_map:
                        phone_ids.append(phone_map.index(phone))
                    else:
                        phone_ids.append(0)  # Unknown phone
                
                # Truncate to max length if needed
                if max_frames and time_frames > max_frames:
                    # Find last phoneme that fits
                    last_idx = 0
                    for i, end in enumerate(phone_ends):
                        if end <= time_frames:
                            last_idx = i
                        else:
                            break
                    
                    # Truncate phoneme data
                    phone_starts = phone_starts[:last_idx+1]
                    phone_ends = phone_ends[:last_idx+1]
                    phone_ids = phone_ids[:last_idx+1]
                    
                    # Ensure last phoneme doesn't go beyond time_frames
                    if phone_ends[-1] > time_frames:
                        phone_ends[-1] = time_frames
                
                # Calculate phoneme durations
                durations = phone_ends - phone_starts
                
                # Store in conditioning dictionary
                conditioning['phoneme_ids'] = torch.tensor(phone_ids).unsqueeze(0)
                conditioning['phoneme_starts'] = torch.tensor(phone_starts).unsqueeze(0)
                conditioning['phoneme_ends'] = torch.tensor(phone_ends).unsqueeze(0)
                conditioning['phoneme_durations'] = torch.tensor(durations).unsqueeze(0)
            
            # Get MIDI pitch information
            if 'MIDI_PITCH' in f:
                midi_pitches = f['MIDI_PITCH'][idx]
                
                # Convert to frame-level representation
                frame_midi = np.zeros(time_frames, dtype=np.int32)
                
                # Fill in frame-level pitch values
                for i, (start, end, pitch) in enumerate(zip(phone_starts, phone_ends, midi_pitches)):
                    # Ensure within bounds
                    start = max(0, min(start, time_frames - 1))
                    end = max(start + 1, min(end, time_frames))
                    
                    # Fill in the range with the pitch value
                    frame_midi[start:end] = pitch
                
                # Store in conditioning dictionary
                conditioning['midi_pitch'] = torch.tensor(frame_midi).unsqueeze(0)
            
            # Get F0 information
            if 'f0_values' in f:
                f0_values = f['f0_values'][idx]
                
                # Truncate to actual length
                f0_values = f0_values[:time_frames]
                
                # Store in conditioning dictionary
                conditioning['f0'] = torch.tensor(f0_values).unsqueeze(0)
            
            return conditioning, time_frames
            
    except Exception as e:
        print(f"Error loading conditioning data: {e}")
        return {}, 0

def mel_to_audio(mel_spec, config):
    """
    Convert mel spectrogram to audio using Griffin-Lim algorithm
    
    Args:
        mel_spec: Mel spectrogram array [mel_bins, time_frames]
        config: Configuration dictionary
        
    Returns:
        Audio waveform
    """
    # Denormalize mel spectrogram
    min_level_db = -80.0
    mel_spec = mel_spec * -min_level_db + min_level_db
    
    # Convert to power spectrum
    power = librosa.db_to_power(mel_spec)
    
    # Use Griffin-Lim to generate audio
    audio = librosa.feature.inverse.mel_to_audio(
        power,
        sr=config['audio']['sample_rate'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        win_length=config['audio']['win_length'],
        center=True,
        power=2.0,
        n_iter=32
    )
    
    return audio

def main():
    parser = argparse.ArgumentParser(description='Generate mel spectrograms from conditioning')
    parser.add_argument('--config', type=str, default='config/conditional_model.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--h5_path', type=str, required=True, help='Path to H5 file with conditioning data')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of sample to use from H5 file')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation (higher = more diverse)')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to generate')
    parser.add_argument('--save_audio', action='store_true', help='Generate and save audio using Griffin-Lim')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = ConditionalMultiBandUNet.load_from_checkpoint(args.checkpoint, config=config)
    model.to(device)
    model.eval()
    
    # Prepare conditioning
    conditioning, time_frames = prep_conditioning_from_file(
        args.h5_path, args.sample_idx, args.max_frames
    )
    
    # Move conditioning to device
    for key in conditioning:
        conditioning[key] = conditioning[key].to(device)
    
    # Generate mel spectrogram
    print(f"Generating mel spectrogram with {time_frames} frames...")
    with torch.no_grad():
        mel = model.generate(
            conditioning=conditioning,
            max_frames=time_frames,
            temperature=args.temperature
        )
    
    # Convert to numpy and remove batch/channel dims
    mel_np = mel[0, 0].cpu().numpy()
    
    # Plot and save the generated mel spectrogram
    plt.figure(figsize=(12, 5))
    plt.imshow(mel_np, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title('Generated Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Bin')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'generated_mel.png'), dpi=300)
    
    # Save the mel spectrogram as numpy array
    np.save(os.path.join(args.output_dir, 'generated_mel.npy'), mel_np)
    
    # Generate audio if requested
    if args.save_audio:
        print("Generating audio using Griffin-Lim algorithm...")
        audio = mel_to_audio(mel_np, config)
        
        # Save audio
        audio_path = os.path.join(args.output_dir, 'generated_audio.wav')
        sf.write(audio_path, audio, config['audio']['sample_rate'])
        print(f"Audio saved to {audio_path}")
    
    print(f"Generation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()