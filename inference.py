import os
import torch
import argparse
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

from utils.utils import load_config, extract_mel_spectrogram_variable_length, normalize_mel_spectrogram, prepare_mel_for_model
from models.combined_model import MultiBandUNetWithHiFiGAN
from models.multi_band_unet import MultiBandUNet
from models.hifi_gan import Generator

def plot_mel_spectrogram(mel, output_path, title=None):
    """Plot mel spectrogram and save to file"""
    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar()
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Bins')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def load_model(model_path, config_path, device):
    """Load the trained model"""
    config = load_config(config_path)
    
    # Check if this is a combined model or separate components
    vocoder_enabled = config.get('vocoder', {}).get('enabled', False)
    
    if vocoder_enabled:
        # Combined model
        model = MultiBandUNetWithHiFiGAN(config).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        
        print(f"Loaded combined model from {model_path}")
        print(f"Epoch: {checkpoint.get('epoch', -1)}, Val Loss: {checkpoint.get('val_loss', -1):.6f}")
    else:
        # Just the U-Net model
        model = MultiBandUNet(config).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        
        print(f"Loaded U-Net model from {model_path}")
        print(f"Epoch: {checkpoint.get('epoch', -1)}, Val Loss: {checkpoint.get('val_loss', -1):.6f}")
    
    model.eval()
    return model, config

def load_separate_models(unet_path, vocoder_path, config_path, device):
    """Load separate U-Net and vocoder models"""
    config = load_config(config_path)
    
    # Load U-Net
    unet = MultiBandUNet(config).to(device)
    unet_checkpoint = torch.load(unet_path, map_location=device)
    unet.load_state_dict(unet_checkpoint)
    unet.eval()
    print(f"Loaded U-Net from {unet_path}")
    
    # Load vocoder
    # Extract vocoder parameters from config
    vocoder_config = config.get('vocoder', {})
    upsample_initial_channel = vocoder_config.get('upsample_initial_channel', 128)
    upsample_rates = vocoder_config.get('upsample_rates', [8, 8, 2, 2])
    upsample_kernel_sizes = vocoder_config.get('upsample_kernel_sizes', [16, 16, 4, 4])
    resblock_kernel_sizes = vocoder_config.get('resblock_kernel_sizes', [3, 7, 11])
    resblock_dilation_sizes = vocoder_config.get('resblock_dilation_sizes', [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    
    vocoder = Generator(
        in_channels=config['model']['mel_bins'],
        upsample_initial_channel=upsample_initial_channel,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes
    ).to(device)
    
    vocoder_checkpoint = torch.load(vocoder_path, map_location=device)
    vocoder.load_state_dict(vocoder_checkpoint)
    vocoder.eval()
    print(f"Loaded vocoder from {vocoder_path}")
    
    return unet, vocoder, config

def process_audio(input_path, output_dir, model, config, device, output_prefix='output'):
    """Process audio file: extract mel, enhance, and vocoder if available"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Extract mel spectrogram
    print(f"Extracting mel spectrogram from {input_path}")
    mel = extract_mel_spectrogram_variable_length(input_path, config)
    
    if mel is None:
        print(f"Failed to extract mel spectrogram from {input_path}")
        return
    
    # Save input mel visualization
    input_mel_path = os.path.join(output_dir, f"{output_prefix}_{base_name}_input_mel.png")
    plot_mel_spectrogram(mel, input_mel_path, title="Input Mel Spectrogram")
    
    # Prepare mel for model
    mel_tensor = prepare_mel_for_model(
        mel, 
        variable_length=config['model'].get('variable_length_mode', False),
        target_bins=config['model'].get('mel_bins', 80)
    ).to(device)
    
    # Check if we're using a combined model or separate U-Net
    is_combined_model = isinstance(model, MultiBandUNetWithHiFiGAN)
    
    with torch.no_grad():
        if is_combined_model:
            # Process with combined model
            outputs = model(mel_tensor)
            mel_output = outputs['mel_output']
            
            # Save processed mel visualization
            output_mel_path = os.path.join(output_dir, f"{output_prefix}_{base_name}_output_mel.png")
            plot_mel_spectrogram(
                mel_output[0, 0].cpu().numpy(), 
                output_mel_path, 
                title="Processed Mel Spectrogram"
            )
            
            # Generate audio if vocoder is enabled
            if model.vocoder_enabled and 'audio_output' in outputs:
                audio_output = outputs['audio_output']
                
                # Save generated audio
                audio_output_path = os.path.join(output_dir, f"{output_prefix}_{base_name}_vocoded.wav")
                audio_np = audio_output[0, 0].cpu().numpy()
                
                # Normalize audio
                audio_np = audio_np / (np.abs(audio_np).max() + 1e-7)
                
                # Save as WAV
                sf.write(
                    audio_output_path, 
                    audio_np, 
                    config['audio']['sample_rate']
                )
                print(f"Saved vocoded audio to {audio_output_path}")
        else:
            # Process with U-Net only
            mel_output = model(mel_tensor)
            
            # Save processed mel visualization
            output_mel_path = os.path.join(output_dir, f"{output_prefix}_{base_name}_output_mel.png")
            plot_mel_spectrogram(
                mel_output[0, 0].cpu().numpy(), 
                output_mel_path, 
                title="Processed Mel Spectrogram"
            )
            
            print(f"Processed mel spectrogram saved to {output_mel_path}")
            print("No vocoder available to generate audio.")

def process_audio_with_separate_models(input_path, output_dir, unet, vocoder, config, device, output_prefix='output'):
    """Process audio using separate U-Net and vocoder models"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # Extract mel spectrogram
    print(f"Extracting mel spectrogram from {input_path}")
    mel = extract_mel_spectrogram_variable_length(input_path, config)
    
    if mel is None:
        print(f"Failed to extract mel spectrogram from {input_path}")
        return
    
    # Save input mel visualization
    input_mel_path = os.path.join(output_dir, f"{output_prefix}_{base_name}_input_mel.png")
    plot_mel_spectrogram(mel, input_mel_path, title="Input Mel Spectrogram")
    
    # Prepare mel for model
    mel_tensor = prepare_mel_for_model(
        mel, 
        variable_length=config['model'].get('variable_length_mode', False),
        target_bins=config['model'].get('mel_bins', 80)
    ).to(device)
    
    with torch.no_grad():
        # Process with U-Net
        mel_output = unet(mel_tensor)
        
        # Save processed mel visualization
        output_mel_path = os.path.join(output_dir, f"{output_prefix}_{base_name}_output_mel.png")
        mel_output_np = mel_output[0, 0].cpu().numpy()
        plot_mel_spectrogram(mel_output_np, output_mel_path, title="Processed Mel Spectrogram")
        
        # Generate audio with vocoder
        # Prepare mel for vocoder: [B, 1, F, T] -> [B, F, T]
        mel_for_vocoder = mel_output.squeeze(1)
        
        # Generate audio
        audio_output = vocoder(mel_for_vocoder)
        
        # Save generated audio
        audio_output_path = os.path.join(output_dir, f"{output_prefix}_{base_name}_vocoded.wav")
        audio_np = audio_output[0, 0].cpu().numpy()
        
        # Normalize audio
        audio_np = audio_np / (np.abs(audio_np).max() + 1e-7)
        
        # Save as WAV
        sf.write(
            audio_output_path, 
            audio_np, 
            config['audio']['sample_rate']
        )
        print(f"Saved vocoded audio to {audio_output_path}")

def main():
    parser = argparse.ArgumentParser(description='Inference with trained model')
    parser.add_argument('--input', type=str, required=True, help='Input audio file or directory')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config_path', type=str, default='config/model_with_vocoder.yaml', help='Path to config file')
    parser.add_argument('--vocoder_path', type=str, default=None, help='Path to separate vocoder checkpoint (if using separate models)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--output_prefix', type=str, default='output', help='Prefix for output files')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load model(s)
    if args.vocoder_path:
        # Load separate models
        unet, vocoder, config = load_separate_models(args.model_path, args.vocoder_path, args.config_path, device)
        use_separate_models = True
    else:
        # Load combined model
        model, config = load_model(args.model_path, args.config_path, device)
        use_separate_models = False
    
    # Process input
    if os.path.isfile(args.input):
        # Single file
        if use_separate_models:
            process_audio_with_separate_models(args.input, args.output_dir, unet, vocoder, config, device, args.output_prefix)
        else:
            process_audio(args.input, args.output_dir, model, config, device, args.output_prefix)
    elif os.path.isdir(args.input):
        # Directory of files
        input_files = [f for f in Path(args.input).glob('**/*.wav')]
        print(f"Found {len(input_files)} WAV files")
        
        for input_file in input_files:
            print(f"Processing {input_file}")
            if use_separate_models:
                process_audio_with_separate_models(str(input_file), args.output_dir, unet, vocoder, config, device, args.output_prefix)
            else:
                process_audio(str(input_file), args.output_dir, model, config, device, args.output_prefix)
    else:
        print(f"Input path {args.input} is neither a file nor a directory")

if __name__ == '__main__':
    main()