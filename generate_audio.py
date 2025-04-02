#!/usr/bin/env python
"""
Quick script to generate audio from mel spectrograms using the trained HiFi-GAN vocoder.
This can be used to convert existing mel spectrograms to audio, or to test the vocoder standalone.
"""

import os
import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from utils.utils import load_config
from models.hifi_gan import Generator

def load_vocoder(checkpoint_path, config, device):
    """Load the HiFi-GAN vocoder from checkpoint"""
    # Get vocoder parameters from config
    vocoder_config = config.get('vocoder', {})
    upsample_initial_channel = vocoder_config.get('upsample_initial_channel', 128)
    upsample_rates = vocoder_config.get('upsample_rates', [8, 8, 2, 2])
    upsample_kernel_sizes = vocoder_config.get('upsample_kernel_sizes', [16, 16, 4, 4])
    resblock_kernel_sizes = vocoder_config.get('resblock_kernel_sizes', [3, 7, 11])
    resblock_dilation_sizes = vocoder_config.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    
    # Create generator
    generator = Generator(
        in_channels=config['model']['mel_bins'],
        upsample_initial_channel=upsample_initial_channel,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        # Extract just the generator part from combined checkpoint
        state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if key.startswith('generator.'):
                state_dict[key.replace('generator.', '')] = value
        
        # Load state dict
        if len(state_dict) > 0:
            generator.load_state_dict(state_dict)
            print(f"Loaded generator from combined checkpoint: {checkpoint_path}")
        else:
            # Try loading as is
            generator.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded generator checkpoint: {checkpoint_path}")
    else:
        # Assume it's a standalone generator checkpoint
        generator.load_state_dict(checkpoint)
        print(f"Loaded standalone generator checkpoint: {checkpoint_path}")
    
    generator.eval()
    return generator

def load_mel_spectrogram(file_path, config):
    """Load mel spectrogram from numpy file"""
    try:
        # Try loading as numpy file
        mel = np.load(file_path)
        
        # Check if shape needs to be corrected (freq bins should be first dimension)
        if mel.shape[0] < mel.shape[1]:
            mel = mel.T
            print(f"Transposed mel spectrogram to shape {mel.shape}")
            
        return mel
    except:
        # If not a numpy file, try loading as audio and extracting mel
        try:
            print(f"Not a numpy file, trying to extract mel from audio: {file_path}")
            
            # Load audio
            y, sr = librosa.load(file_path, sr=config['audio']['sample_rate'])
            
            # Extract mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=config['audio']['n_fft'],
                hop_length=config['audio']['hop_length'],
                win_length=config['audio']['win_length'],
                n_mels=config['audio']['n_mels'],
                fmin=config['audio']['fmin'],
                fmax=config['audio']['fmax'],
            )
            
            # Convert to dB scale
            mel = librosa.power_to_db(mel, ref=np.max)
            
            # Normalize to [0, 1]
            mel = (mel - mel.min()) / (mel.max() - mel.min())
            
            return mel
        except Exception as e:
            print(f"Error loading mel spectrogram from {file_path}: {e}")
            return None

def generate_audio(mel, generator, config, device):
    """Generate audio from mel spectrogram using the vocoder"""
    # Prepare mel for the generator
    # Add batch dimension if needed
    if mel.ndim == 2:
        mel = mel[np.newaxis, ...]
    
    # Convert to tensor
    mel_tensor = torch.FloatTensor(mel).to(device)
    
    # Generate audio
    with torch.no_grad():
        audio = generator(mel_tensor)
    
    # Convert to numpy and remove batch dimension
    audio = audio.squeeze().cpu().numpy()
    
    # Ensure audio is float32 (soundfile doesn't support float16)
    audio = audio.astype(np.float32)
    
    # Normalize
    audio = audio / (np.abs(audio).max() + 1e-7)
    
    return audio

def main():
    parser = argparse.ArgumentParser(description='Generate audio from mel spectrograms using HiFi-GAN')
    parser.add_argument('--input', type=str, required=True, help='Input mel spectrogram or directory of mel spectrograms')
    parser.add_argument('--output_dir', type=str, default='vocoded_audio', help='Output directory for generated audio')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to HiFi-GAN checkpoint')
    parser.add_argument('--config', type=str, default='config/model_with_vocoder.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--ext', type=str, default='.npy', help='Extension of mel spectrogram files if input is a directory')
    parser.add_argument('--plot', action='store_true', help='Plot and save mel spectrograms')
    parser.add_argument('--sample_rate', type=int, default=None, help='Override sample rate from config')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override sample rate if specified
    if args.sample_rate:
        config['audio']['sample_rate'] = args.sample_rate
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    if args.plot:
        mel_plot_dir = os.path.join(args.output_dir, 'mel_plots')
        os.makedirs(mel_plot_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load vocoder
    generator = load_vocoder(args.checkpoint, config, device)
    
    # Process input
    if os.path.isfile(args.input):
        # Single file
        mel = load_mel_spectrogram(args.input, config)
        if mel is None:
            print(f"Failed to load mel spectrogram from {args.input}")
            return
        
        # Plot mel if requested
        if args.plot:
            output_plot_path = os.path.join(mel_plot_dir, f"{Path(args.input).stem}_mel.png")
            plt.figure(figsize=(10, 4))
            plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar()
            plt.title(f"Mel Spectrogram: {Path(args.input).name}")
            plt.xlabel('Time Frames')
            plt.ylabel('Mel Bins')
            plt.tight_layout()
            plt.savefig(output_plot_path)
            plt.close()
            print(f"Saved mel plot to {output_plot_path}")
        
        # Generate audio
        audio = generate_audio(mel, generator, config, device)
        
        # Save audio
        output_path = os.path.join(args.output_dir, f"{Path(args.input).stem}_vocoded.wav")
        sf.write(output_path, audio, config['audio']['sample_rate'])
        print(f"Generated audio saved to {output_path}")
    
    elif os.path.isdir(args.input):
        # Directory of files
        input_files = list(Path(args.input).glob(f"**/*{args.ext}"))
        if len(input_files) == 0:
            # Try .npy files if no files with specified extension
            input_files = list(Path(args.input).glob("**/*.npy"))
        
        if len(input_files) == 0:
            # Try audio files if no numpy files
            input_files = list(Path(args.input).glob("**/*.wav"))
            input_files.extend(list(Path(args.input).glob("**/*.mp3")))
            input_files.extend(list(Path(args.input).glob("**/*.flac")))
        
        print(f"Found {len(input_files)} input files")
        
        # Process each file
        for input_file in tqdm(input_files, desc="Generating audio"):
            mel = load_mel_spectrogram(str(input_file), config)
            if mel is None:
                print(f"Failed to load mel spectrogram from {input_file}")
                continue
            
            # Plot mel if requested
            if args.plot:
                output_plot_path = os.path.join(mel_plot_dir, f"{input_file.stem}_mel.png")
                plt.figure(figsize=(10, 4))
                plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
                plt.colorbar()
                plt.title(f"Mel Spectrogram: {input_file.name}")
                plt.xlabel('Time Frames')
                plt.ylabel('Mel Bins')
                plt.tight_layout()
                plt.savefig(output_plot_path)
                plt.close()
            
            # Generate audio
            audio = generate_audio(mel, generator, config, device)
            
            # Save audio
            output_path = os.path.join(args.output_dir, f"{input_file.stem}_vocoded.wav")
            sf.write(output_path, audio, config['audio']['sample_rate'])
    else:
        print(f"Input path {args.input} is neither a file nor a directory")

if __name__ == '__main__':
    main()