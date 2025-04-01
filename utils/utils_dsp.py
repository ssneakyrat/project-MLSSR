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
