import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_mel_spectrogram(wav_path, config):
    """
    Extract mel spectrogram from a wav file.
    
    Args:
        wav_path (str): Path to the wav file
        config (dict): Configuration dictionary containing audio parameters
        
    Returns:
        numpy.ndarray: Mel spectrogram
    """
    try:
        # Load audio file
        y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=config['audio']['n_fft'],
            hop_length=config['audio']['hop_length'],
            win_length=config['audio']['win_length'],
            n_mels=config['audio']['n_mels'],
            fmin=config['audio']['fmin'],
            fmax=config['audio']['fmax'],
        )
        
        # Convert to log scale (dB)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        print(f"Extracted mel spectrogram with shape: {mel_spec.shape}")
        return mel_spec
        
    except Exception as e:
        print(f"Error extracting mel spectrogram from {wav_path}: {e}")
        return None
        
def extract_f0(wav_path, config):
    """
    Extract fundamental frequency (F0) from a wav file using PYIN algorithm,
    which is considered one of the best methods for monophonic pitch tracking.
    
    Args:
        wav_path (str): Path to the wav file
        config (dict): Configuration dictionary containing audio parameters
        
    Returns:
        numpy.ndarray: F0 contour
    """
    try:
        # Load audio file
        y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
        
        # Extract F0 using PYIN algorithm (Probabilistic YIN)
        # This is generally more accurate than simpler methods
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=config.get('audio', {}).get('f0_min', 50),  # Default min F0: 50 Hz
            fmax=config.get('audio', {}).get('f0_max', 600),  # Default max F0: 600 Hz
            sr=sr,
            hop_length=config['audio']['hop_length']
        )
        
        # Replace NaN values with zeros (for unvoiced regions)
        f0 = np.nan_to_num(f0)
        
        print(f"Extracted F0 contour with shape: {f0.shape}")
        return f0
        
    except Exception as e:
        print(f"Error extracting F0 from {wav_path}: {e}")
        return None

def plot_mel_and_f0(mel_spec, f0, output_path):
    """
    Plot mel spectrogram with F0 contour overlay and save to an image file.
    
    Args:
        mel_spec (numpy.ndarray): Mel spectrogram
        f0 (numpy.ndarray): F0 contour
        output_path (str): Path to save the image file
    """
    try:
        plt.figure(figsize=(12, 8))
        
        # Create a new axis for the spectrogram
        plt.subplot(2, 1, 1)
        
        # Plot mel spectrogram
        librosa.display.specshow(
            mel_spec,
            y_axis='mel',
            x_axis='time',
            sr=22050,
            hop_length=256,
            cmap='viridis'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        
        # Create a new axis for both mel and F0
        plt.subplot(2, 1, 2)
        
        # Plot mel spectrogram again
        librosa.display.specshow(
            mel_spec,
            y_axis='mel',
            x_axis='time',
            sr=22050,
            hop_length=256,
            cmap='viridis'
        )
        
        # Get time points for F0
        times = np.arange(len(f0)) * 256 / 22050
        
        # Create a twin axis for F0
        ax2 = plt.gca().twinx()
        ax2.plot(times, f0, 'r-', linewidth=2, alpha=0.7, label='F0')
        ax2.set_ylabel('F0 (Hz)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim([0, 600])  # Adjust this range based on expected F0 values
        
        plt.title('Mel Spectrogram with F0 Contour')
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the figure
        plt.savefig(output_path)
        plt.close()
        
        print(f"Plotted mel spectrogram and F0 contour to {output_path}")
        
    except Exception as e:
        print(f"Error plotting mel spectrogram and F0 contour: {e}")
        return None