import yaml
import numpy as np
import librosa
import os

def load_config(config_path="config/default.yaml"):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Configuration file {config_path} not found. Using default configuration.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        return {}

def extract_mel_spectrogram(wav_path, config):
    try:
        y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
        
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
        
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec
        
    except Exception as e:
        print(f"Error extracting mel spectrogram from {wav_path}: {e}")
        return None

def extract_f0(wav_path, config):
    try:
        y, sr = librosa.load(wav_path, sr=config['audio']['sample_rate'])
        
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=config.get('audio', {}).get('f0_min', 50),
            fmax=config.get('audio', {}).get('f0_max', 600),
            sr=sr,
            hop_length=config['audio']['hop_length']
        )
        
        f0 = np.nan_to_num(f0)
        return f0
        
    except Exception as e:
        print(f"Error extracting F0 from {wav_path}: {e}")
        return None

def normalize_mel_spectrogram(mel_spec):
    if np.min(mel_spec) < 0:
        mel_spec = np.clip(mel_spec, -80.0, 0.0)
        mel_spec = (mel_spec + 80.0) / 80.0
    else:
        max_val = np.max(mel_spec)
        min_val = np.min(mel_spec)
        if max_val > min_val:
            mel_spec = (mel_spec - min_val) / (max_val - min_val)
    
    return mel_spec

def pad_or_truncate_mel(mel_spec, target_length=128, target_bins=80):
    curr_bins, curr_length = mel_spec.shape
    
    resized_mel = np.zeros((target_bins, target_length), dtype=mel_spec.dtype)
    
    freq_bins_to_copy = min(curr_bins, target_bins)
    time_frames_to_copy = min(curr_length, target_length)
    
    resized_mel[:freq_bins_to_copy, :time_frames_to_copy] = mel_spec[:freq_bins_to_copy, :time_frames_to_copy]
    
    return resized_mel

def prepare_mel_for_model(mel_spec, target_length=128, target_bins=80):
    import torch
    
    if mel_spec.shape[0] > mel_spec.shape[1]:
        mel_spec = mel_spec.T
    
    mel_spec = normalize_mel_spectrogram(mel_spec)
    mel_spec = pad_or_truncate_mel(mel_spec, target_length, target_bins)
    
    mel_tensor = torch.from_numpy(mel_spec).float()
    mel_tensor = mel_tensor.permute(1, 0).unsqueeze(0)
    
    return mel_tensor