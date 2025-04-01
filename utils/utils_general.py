
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

def load_config(config_path="config/default.yaml"):
    """
    Load configuration from a YAML file.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found. Using default configuration.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return {}
    
import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display

def plot_alignment(mel_spec, f0, phonemes, output_path, config=None, time_scale=None):
    """
    Generate a visualization showing the alignment between phonemes and audio features.
    
    Args:
        mel_spec (numpy.ndarray): Mel spectrogram with shape (n_mels, time)
        f0 (numpy.ndarray): F0 contour with shape (time,)
        phonemes (list): List of phoneme tuples (start_time, end_time, phoneme)
        output_path (str): Path to save the visualization
        config (dict, optional): Configuration dictionary
        time_scale (float, optional): Scale factor to convert phoneme timings to seconds.
                                    If None, will attempt to auto-detect.
        
    Returns:
        str: Path to the saved visualization file
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Default audio parameters if config is not provided
    sample_rate = config.get('audio', {}).get('sample_rate', 22050) if config else 22050
    hop_length = config.get('audio', {}).get('hop_length', 256) if config else 256
    fmin = config.get('audio', {}).get('fmin', 0) if config else 0
    fmax = config.get('audio', {}).get('fmax', 8000) if config else 8000
    
    # Calculate times for the mel spectrogram
    n_frames = mel_spec.shape[1]
    times = librosa.times_like(np.arange(n_frames), sr=sample_rate, hop_length=hop_length)
    
    # Calculate F0 times
    f0_times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length)
    
    # Calculate the audio duration
    audio_duration = times[-1] if times.size > 0 else 0
    
    # Auto-detect time scale if not provided
    if time_scale is None:
        if phonemes:
            # Get the last phoneme end time
            last_phoneme_end = max(end for _, end, _ in phonemes)
            
            # Estimate time scale based on audio duration
            if last_phoneme_end > 0:
                time_scale = audio_duration / last_phoneme_end
                print(f"Auto-detected time scale: {time_scale}")
            else:
                # Default to HTK format (100ns units)
                time_scale = 1e-7
                print(f"Using default time scale: {time_scale}")
        else:
            # Default to HTK format if no phonemes
            time_scale = 1e-7
    
    # Create a figure with three subplots (mel, f0, phoneme alignment)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), 
                                       sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot mel spectrogram
    img = librosa.display.specshow(
        mel_spec, 
        x_axis='time',
        y_axis='mel', 
        sr=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        ax=ax1
    )
    ax1.set_title('Mel Spectrogram')
    fig.colorbar(img, ax=ax1, format='%+2.0f dB')
    ax1.grid(axis='x', color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Plot F0 contour
    ax2.plot(f0_times, f0, color='b')
    ax2.set_ylabel('F0 (Hz)')
    ax2.set_title('Fundamental Frequency (F0)')
    ax2.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Generate a color map for phonemes
    unique_phonemes = sorted(set(phone for _, _, phone in phonemes))
    cmap = plt.cm.get_cmap('tab20', len(unique_phonemes))
    phoneme_colors = {phone: cmap(i) for i, phone in enumerate(unique_phonemes)}
    
    # Plot phoneme alignment
    ax3.set_ylim(0, 1)
    ax3.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Plot phoneme segments
    for start, end, phone in phonemes:
        start_sec = start * time_scale
        end_sec = end * time_scale
        
        # Skip segments outside the audio range
        if start_sec > audio_duration:
            continue
            
        # Clip end time to the audio range
        end_sec = min(end_sec, audio_duration)
        
        # Draw a colored rectangle for the phoneme segment
        rect = plt.Rectangle(
            (start_sec, 0.2), end_sec - start_sec, 0.6, 
            facecolor=phoneme_colors[phone], alpha=0.7, edgecolor='black', linewidth=0.5
        )
        ax3.add_patch(rect)
        
        # Add phoneme label at the center of the segment if wide enough
        center = (start_sec + end_sec) / 2
        width = end_sec - start_sec
        
        # Only add text if the segment is wide enough
        if width > audio_duration / 100:  # Skip very narrow segments
            ax3.text(center, 0.5, phone, 
                    horizontalalignment='center', verticalalignment='center',
                    color='black', fontsize=8, fontweight='bold')
    
    # Create a custom legend for phoneme colors
    if len(unique_phonemes) <= 20:  # Only add legend if not too many phonemes
        legend_elements = [plt.Rectangle((0,0),1,1, color=phoneme_colors[p], label=p) 
                          for p in unique_phonemes]
        ax3.legend(handles=legend_elements, loc='upper center', 
                  bbox_to_anchor=(0.5, -0.15), ncol=min(10, len(unique_phonemes)), 
                  fontsize=8, frameon=True)
    
    # Set labels for phoneme alignment
    ax3.set_title('Phoneme Alignment')
    ax3.set_xlabel('Time (s)')
    ax3.set_yticks([])  # No y-ticks for phoneme display
    
    # Set the x-axis limits
    plt.xlim(0, audio_duration)
    
    # Add title
    plt.suptitle("Audio Features and Phoneme Alignment", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Alignment visualization saved to {output_path}")
    
    return output_path