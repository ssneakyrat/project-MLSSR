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
    Generate a visualization showing the alignment between phonemes, audio features, and MIDI notes,
    with the mel spectrogram as a separate plot.
    
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
    
    # Create a figure with two subplots - cut size in half but increase element sizes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 2), sharex=True, 
                                  gridspec_kw={'height_ratios': [2, 3]})
                                  
    # Set font sizes larger for better readability in small image
    plt.rcParams.update({
        'font.size': 4,
        'axes.titlesize': 5,
        'axes.labelsize': 4,
        'xtick.labelsize': 3,
        'ytick.labelsize': 3,
        'legend.fontsize': 3,
        'figure.titlesize': 6
    })
    
    # Plot the mel spectrogram (without colorbar) - with increased visibility
    img = librosa.display.specshow(
        mel_spec, 
        x_axis='time',
        y_axis='mel', 
        sr=sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        ax=ax1,
        cmap='viridis'
    )
    
    # Title for the mel spectrogram with increased size
    ax1.set_title('Mel Spectrogram', fontsize=5)
    
    # Title for the mel spectrogram
    ax1.set_title('Mel Spectrogram', fontsize=16)
    
    # ===================
    # Plot 2: F0, MIDI, and Phoneme Alignment
    # ===================
    
    # Plot F0 contour with thicker line
    ax2.plot(f0_times, f0, color='red', linewidth=0.8, label='F0')
    ax2.set_ylabel('F0 (Hz)', color='red', fontsize=4)
    ax2.tick_params(axis='y', labelcolor='red', labelsize=3)
    ax2.set_ylim(f0_min, f0_max)
    
    # Generate a color map for phonemes
    unique_phonemes = sorted(set(phone for _, _, phone in phonemes))
    cmap = plt.cm.get_cmap('tab20', len(unique_phonemes))
    phoneme_colors = {phone: cmap(i) for i, phone in enumerate(unique_phonemes)}
    
    # Add phoneme segments as rectangles at the bottom of the second plot
    y_min, y_max = ax2.get_ylim()
    rect_height = (y_max - y_min) * 0.1  # Height for phoneme rectangles
    rect_bottom = y_min  # Position at the bottom of the plot
    
    # Draw a line to separate the phoneme area
    ax2.axhline(y=rect_bottom + rect_height, color='black', linestyle='-', alpha=0.5, linewidth=0.3)
    
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
            (start_sec, rect_bottom), end_sec - start_sec, rect_height, 
            facecolor=phoneme_colors[phone], alpha=0.7, edgecolor='black', linewidth=0.2
        )
        ax2.add_patch(rect)
        
        # Add phoneme label at the center of the segment if wide enough
        center = (start_sec + end_sec) / 2
        width = end_sec - start_sec
        
        # Only add text if the segment is wide enough
        if width > audio_duration / 50:  # Stricter threshold for small image
            ax2.text(center, rect_bottom + rect_height/2, phone, 
                   horizontalalignment='center', verticalalignment='center',
                   color='black', fontsize=2, fontweight='bold')
    
    # Add MIDI notes to the second subplot if available - with thinner lines
    if midi_notes is not None:
        # Get valid MIDI notes
        midi_valid = midi_notes[midi_notes > 0]
        if len(midi_valid) > 0:
            midi_min = np.floor(np.min(midi_valid) - 3)
            midi_max = np.ceil(np.max(midi_valid) + 3)
            
            # Create a secondary y-axis for MIDI notes
            ax3 = ax2.twinx()
            ax3.plot(f0_times, midi_notes, color='purple', linewidth=0.6, alpha=0.7, label='MIDI')
            ax3.set_ylabel('MIDI Note', color='purple', fontsize=4)
            ax3.tick_params(axis='y', labelcolor='purple', labelsize=3)
            ax3.set_ylim(midi_min, midi_max)
            
            # Add horizontal grid lines at octave intervals
            for i in range(int(midi_min), int(midi_max) + 1, 12):  # Every octave (C notes)
                ax3.axhline(y=i, color='purple', linestyle='--', alpha=0.3, linewidth=0.2)
                ax3.text(audio_duration * 1.01, i, get_note_name(i), 
                       color='purple', fontsize=2, verticalalignment='center')
            
            # Plot phoneme MIDI notes
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
                    ax3.plot([start_sec, end_sec], [midi, midi], 
                           color='yellow', linewidth=0.8, alpha=0.7)
                    
                    # Add note name at the center of the segment if wide enough
                    center = (start_sec + end_sec) / 2
                    width = end_sec - start_sec
                    
                    # Only add text if the segment is wide enough
                    if width > audio_duration / 25:  # Stricter threshold for small image
                        note_name = get_note_name(int(round(midi)))
                        ax3.text(center, midi + 0.5, note_name,
                               horizontalalignment='center', verticalalignment='bottom',
                               color='yellow', fontsize=2, fontweight='bold', 
                               bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=0.5))
    
    # Create legends for the second subplot with smaller size
    # Phoneme legend - show fewer examples for better readability
    if len(unique_phonemes) <= 10:  # Limit to fewer phonemes for small plot
        max_legend_phonemes = min(6, len(unique_phonemes))  # At most 6 phonemes in legend
        # Take a sample of phonemes if there are too many
        legend_phonemes = unique_phonemes[:max_legend_phonemes]
        
        phoneme_patches = [plt.Rectangle((0,0), 1, 1, facecolor=phoneme_colors[p], alpha=0.7, 
                                        edgecolor='black', label=p) 
                         for p in legend_phonemes]
        phoneme_legend = ax2.legend(handles=phoneme_patches, loc='lower center',
                                  bbox_to_anchor=(0.5, -0.3), ncol=min(3, max_legend_phonemes),
                                  fontsize=2, frameon=True, title='Phonemes', title_fontsize=3)
        
        # Add the legend to the figure
        fig.add_artist(phoneme_legend)
    
    # F0 and MIDI legend - with smaller size
    line_legends = [plt.Line2D([0], [0], color='red', lw=0.8, label='F0')]
    if midi_notes is not None and len(midi_valid) > 0:
        line_legends.append(plt.Line2D([0], [0], color='purple', lw=0.6, label='MIDI'))
        line_legends.append(plt.Line2D([0], [0], color='yellow', lw=0.8, label='Ph-MIDI'))
        
    ax2.legend(handles=line_legends, loc='upper right', fontsize=2, frameon=True)
    
    # Title for the alignment plot
    ax2.set_title('F0, MIDI Notes, and Phoneme Alignment', fontsize=5)
    
    # Add grid for the second subplot
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Set x-axis limits
    plt.xlim(0, audio_duration)
    plt.xlabel('Time (s)', fontsize=4)
    
    # Add overall title
    fig.suptitle("Audio Features and Phoneme Alignment", fontsize=6)
    
    # Adjust layout with very tight spacing
    plt.tight_layout(rect=[0, 0, 1, 0.96], pad=0.1, h_pad=0.5, w_pad=0.1)
    
    if return_bytes:
        # Save to in-memory buffer instead of file
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight')
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