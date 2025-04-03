import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse
import librosa
import os
from matplotlib.colors import LinearSegmentedColormap

def visualize_phoneme_midi_pitch(h5_file_path, sample_idx=0):
    """
    Visualize the MIDI pitch estimates for phonemes in a sample along with aligned F0.
    
    Args:
        h5_file_path: Path to the H5 file containing the preprocessed data
        sample_idx: Index of the sample to visualize
    """
    with h5py.File(h5_file_path, 'r') as f:
        # Get file ID for this sample
        file_ids = f['file_ids'][:]
        if sample_idx >= len(file_ids):
            print(f"Sample index {sample_idx} out of range (max: {len(file_ids)-1})")
            return
            
        file_id = file_ids[sample_idx]
        print(f"Visualizing file: {file_id}")
        
        # Get sample rate and hop length
        data_key = list(f.keys())[0]
        for key in f.keys():
            if key.startswith('mel') or 'spectrogram' in key.lower():
                data_key = key
                break
                
        sample_rate = f[data_key].attrs['sample_rate']
        hop_length = f[data_key].attrs['hop_length']
        
        # Get actual length of this sample
        if 'lengths' in f:
            length = f['lengths'][sample_idx]
        else:
            length = f[data_key][sample_idx].shape[1]  # Assume second dimension is time
        
        # Get mel spectrogram
        mel_spec = f[data_key][sample_idx]
        if len(mel_spec.shape) == 3:  # If it has a channel dimension
            mel_spec = mel_spec[0]  # Take the first channel
            
        # Get aligned F0 data
        f0_aligned = None
        if 'f0_aligned' in f:
            f0_aligned = f['f0_aligned'][sample_idx][:length]  # Only take valid length
        
        # Get phoneme data
        has_phoneme_data = 'phoneme_starts' in f and 'phoneme_midi_pitches' in f
        
        if has_phoneme_data:
            phoneme_starts = f['phoneme_starts'][sample_idx]
            phoneme_ends = f['phoneme_ends'][sample_idx]
            phoneme_texts = f['phoneme_texts'][sample_idx]
            midi_pitches = f['phoneme_midi_pitches'][sample_idx]
            
            # Convert time from samples to seconds
            phoneme_starts_sec = phoneme_starts / sample_rate
            phoneme_ends_sec = phoneme_ends / sample_rate
        
        # Create figure with appropriate number of subplots
        if has_phoneme_data:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # 1. Plot mel spectrogram
        # Create a custom colormap from viridis
        viridis = plt.cm.viridis
        newcolors = viridis(np.linspace(0, 1, 256))
        newcolors[:25, :] = np.array([0, 0, 0, 1.0])  # Make the lowest values black
        mel_cmap = LinearSegmentedColormap.from_list('mel_colormap', newcolors)
        
        mel_img = ax1.imshow(
            mel_spec[:, :length], 
            aspect='auto', 
            origin='lower',
            cmap=mel_cmap,
            extent=[0, length * hop_length / sample_rate, 0, mel_spec.shape[0]]
        )
        fig.colorbar(mel_img, ax=ax1, label='dB')
        ax1.set_title(f'Mel Spectrogram - {file_id}')
        ax1.set_ylabel('Mel Bin')
        
        # Add phoneme boundaries if available
        if has_phoneme_data:
            for start, end, text in zip(phoneme_starts_sec, phoneme_ends_sec, phoneme_texts):
                ax1.axvline(x=start, color='r', linestyle='--', alpha=0.3)
                mid_point = (start + end) / 2
                ax1.text(mid_point, mel_spec.shape[0] * 0.9, text, 
                        ha='center', fontsize=8, 
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # 2. Plot F0 contour
        if f0_aligned is not None:
            # Convert frame indices to time in seconds
            time_axis = np.arange(len(f0_aligned)) * hop_length / sample_rate
            ax2.plot(time_axis, f0_aligned, 'b-', alpha=0.7, label='F0 (Hz)')
            ax2.set_title(f'F0 Contour')
            ax2.set_ylabel('Frequency (Hz)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add phoneme boundaries if available
            if has_phoneme_data:
                for start, end in zip(phoneme_starts_sec, phoneme_ends_sec):
                    ax2.axvline(x=start, color='r', linestyle='--', alpha=0.3)
        else:
            ax2.set_visible(False)
            
        # 3. Plot MIDI pitch if phoneme data is available
        if has_phoneme_data:
            for i, (start, end, text, pitch) in enumerate(zip(
                phoneme_starts_sec, phoneme_ends_sec, phoneme_texts, midi_pitches)):
                
                # Skip unvoiced phonemes (midi pitch 0)
                if pitch == 0:
                    color = 'gray'
                    alpha = 0.3
                    note_name = "N/A"
                else:
                    color = 'blue'
                    alpha = 0.6
                    note_name = librosa.midi_to_note(pitch)
                
                # Draw a horizontal line for each phoneme
                ax3.plot([start, end], [pitch, pitch], '-', color=color, alpha=alpha, linewidth=2)
                
                # Add a marker at the center
                mid_point = (start + end) / 2
                ax3.plot(mid_point, pitch, 'o', color=color, alpha=alpha+0.2)
                
                # Add text label with phoneme and note name
                ax3.text(mid_point, pitch + 1, f"{text}\n{note_name}", 
                        ha='center', va='bottom', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            
            # Format the MIDI pitch plot
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('MIDI Pitch')
            ax3.set_title('Phoneme MIDI Pitch Estimates')
            ax3.grid(True, alpha=0.3)
            
            # Set y-axis range for MIDI plot (showing common vocal range)
            ax3.set_ylim(36, 84)  # C2 to C6
            
            # Add piano roll reference lines
            for midi_note in range(36, 85, 12):  # C2 to C6
                ax3.axhline(y=midi_note, color='lightgray', linestyle='-', alpha=0.5)
                ax3.text(0, midi_note + 0.5, librosa.midi_to_note(midi_note), fontsize=8)
        
        # Make sure x-axis covers the full duration
        plt.xlim(0, length * hop_length / sample_rate)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir = os.path.dirname(h5_file_path)
        output_path = os.path.join(output_dir, f'phoneme_midi_pitch_{file_id}.png')
        plt.savefig(output_path, dpi=300)
        print(f"Saved visualization to {output_path}")
        
        plt.show()
        
        # Print the phoneme data if available
        if has_phoneme_data:
            print("\nPhoneme MIDI Pitch Data:")
            print("-" * 60)
            print(f"{'Phoneme':<10} {'Start (s)':<10} {'End (s)':<10} {'MIDI Pitch':<10} {'Note':<10}")
            print("-" * 60)
            
            for text, start, end, pitch in zip(phoneme_texts, phoneme_starts_sec, phoneme_ends_sec, midi_pitches):
                note_name = librosa.midi_to_note(pitch) if pitch > 0 else "N/A"
                print(f"{text:<10} {start:<10.2f} {end:<10.2f} {pitch:<10.2f} {note_name:<10}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize phoneme MIDI pitch estimates')
    parser.add_argument('--h5_file', type=str, required=True, help='Path to H5 file')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to visualize')
    
    args = parser.parse_args()
    
    visualize_phoneme_midi_pitch(args.h5_file, args.sample_idx)