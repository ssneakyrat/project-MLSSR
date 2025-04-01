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

def get_note_name(midi_number):
    """
    Convert MIDI note number to note name (e.g., C4, A#3).
    
    Args:
        midi_number (int): MIDI note number
        
    Returns:
        str: Note name
    """
    if midi_number <= 0:
        return "Rest"
        
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = note_names[midi_number % 12]
    return f"{note}{octave}"

def plot_alignment(mel_spec, f0, phonemes, output_path=None, config=None, time_scale=None, 
                  phoneme_midi=None, midi_notes=None, return_bytes=False):
    """
    Generate a visualization showing the alignment between phonemes, audio features, and MIDI notes.
    
    Args:
        mel_spec (numpy.ndarray): Mel spectrogram with shape (n_mels, time)
        f0 (numpy.ndarray): F0 contour with shape (time,)
        phonemes (list): List of phoneme tuples (start_time, end_time, phoneme)
        output_path (str, optional): Path to save the visualization. Not used if return_bytes=True.
        config (dict, optional): Configuration dictionary
        time_scale (float, optional): Scale factor to convert phoneme timings to seconds.
                                    If None, will attempt to auto-detect.
        phoneme_midi (list, optional): List of phoneme tuples with MIDI notes 
                                      (start_time, end_time, phoneme, midi_note)
        midi_notes (numpy.ndarray, optional): Frame-level MIDI note values
        return_bytes (bool, optional): If True, return image bytes instead of saving to disk.
        
    Returns:
        str or bytes: Path to the saved visualization file or image bytes if return_bytes=True
    """
    import matplotlib
    if return_bytes:
        # Use Agg backend when returning bytes (no GUI needed)
        matplotlib.use('Agg')
        
    import matplotlib.pyplot as plt
    import numpy as np
    import librosa
    import librosa.display
    import io
    
    # Ensure the output directory exists if saving to disk
    if output_path and not return_bytes:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Default audio parameters if config is not provided
    sample_rate = config.get('audio', {}).get('sample_rate', 22050) if config else 22050
    hop_length = config.get('audio', {}).get('hop_length', 256) if config else 256
    fmin = config.get('audio', {}).get('fmin', 0) if config else 0
    fmax = config.get('audio', {}).get('fmax', 8000) if config else 8000
    f0_min = config.get('audio', {}).get('f0_min', 50) if config else 50
    f0_max = config.get('audio', {}).get('f0_max', 600) if config else 600
    
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
    
    # Determine number of subplots based on available data
    n_plots = 3  # Mel, F0, Phonemes
    if midi_notes is not None:
        n_plots += 1  # Add MIDI plot
    
    # Create a figure with appropriate subplots
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), 
                           sharex=True, 
                           gridspec_kw={'height_ratios': [3] + [1] * (n_plots - 1)})
    
    # First layer: Mel spectrogram
    ax1 = axes[0]
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
    ax1.grid(axis='x', color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Second layer: F0 contour
    ax2 = axes[1]
    ax2.plot(f0_times, f0, color='b')
    ax2.set_ylabel('F0 (Hz)')
    ax2.set_title('Fundamental Frequency (F0)')
    ax2.set_ylim(f0_min - 10, f0_max + 10)
    ax2.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Get MIDI times if available
    if midi_notes is not None:
        midi_times = f0_times  # Same time scale as F0
    
    # Third layer: Phoneme alignment
    ax3 = axes[2]
    ax3.set_ylim(0, 1)
    ax3.set_title('Phoneme Alignment')
    ax3.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.set_yticks([])  # No y-ticks for phoneme display
    
    # Generate a color map for phonemes
    unique_phonemes = sorted(set(phone for _, _, phone in phonemes))
    cmap = plt.cm.get_cmap('tab20', len(unique_phonemes))
    phoneme_colors = {phone: cmap(i) for i, phone in enumerate(unique_phonemes)}
    
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
    
    # Create a legend for phonemes (only if not too many)
    if len(unique_phonemes) <= 20:
        legend_elements = [plt.Rectangle((0,0),1,1, color=phoneme_colors[p], label=p) 
                         for p in unique_phonemes]
        ax3.legend(handles=legend_elements, loc='upper center', 
                 bbox_to_anchor=(0.5, -0.15), ncol=min(10, len(unique_phonemes)), 
                 fontsize=8, frameon=True)
    
    # Fourth layer (if available): MIDI notes
    if midi_notes is not None:
        ax4 = axes[3]
        
        # Plot frame-level MIDI notes (if available)
        if midi_notes is not None:
            # Get the valid range of MIDI notes (exclude zeros)
            midi_valid = midi_notes[midi_notes > 0]
            y_min = max(0, np.floor(np.min(midi_valid) - 3)) if len(midi_valid) > 0 else 36
            y_max = np.ceil(np.max(midi_valid) + 3) if len(midi_valid) > 0 else 84
            
            # Plot the MIDI notes
            ax4.plot(midi_times, midi_notes, color='purple', linewidth=1.5)
            ax4.set_ylabel('MIDI Note')
            ax4.set_title('MIDI Notes')
            ax4.set_ylim(y_min, y_max)
            
            # Add grid lines at each MIDI note
            for i in range(int(y_min), int(y_max) + 1):
                ax4.axhline(y=i, color='lightgray', linestyle='-', alpha=0.5, linewidth=0.5)
            
            # Label some of the MIDI note lines with note names
            note_labels = []
            for i in range(int(y_min), int(y_max) + 1):
                if i % 12 == 0:  # Label C notes
                    note_labels.append(i)
                    ax4.text(-0.01, i, get_note_name(i), 
                             horizontalalignment='right', verticalalignment='center',
                             fontsize=8, transform=ax4.get_yaxis_transform())
        
        # Plot phoneme-level MIDI notes (if available)
        if phoneme_midi is not None:
            for start, end, phone, midi in phoneme_midi:
                if midi <= 0:  # Skip unvoiced phonemes
                    continue
                    
                start_sec = start * time_scale
                end_sec = end * time_scale
                
                # Skip segments outside the audio range
                if start_sec > audio_duration:
                    continue
                    
                # Clip end time to the audio range
                end_sec = min(end_sec, audio_duration)
                
                # Draw a horizontal line for the MIDI note
                ax4.plot([start_sec, end_sec], [midi, midi], 
                         color='red', linewidth=3, alpha=0.7)
                
                # Add note name at the center of the segment if wide enough
                center = (start_sec + end_sec) / 2
                width = end_sec - start_sec
                
                # Only add text if the segment is wide enough
                if width > audio_duration / 50:  # Skip very narrow segments
                    note_name = get_note_name(int(round(midi)))
                    ax4.text(center, midi + 0.5, note_name,
                             horizontalalignment='center', verticalalignment='bottom',
                             color='darkred', fontsize=8, fontweight='bold')
        
        ax4.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Set the x-axis limits
    plt.xlim(0, audio_duration)
    plt.xlabel('Time (s)')
    
    # Add title
    plt.suptitle("Audio Features, Phoneme Alignment, and MIDI Notes", fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if return_bytes:
        # Save to in-memory buffer instead of file
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        
        print("Plot generated as bytes")
        return buf.getvalue()
    else:
        # Save to file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Alignment visualization saved to {output_path}")
        return output_path