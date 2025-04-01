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
        
        #print(f"Extracted mel spectrogram with shape: {mel_spec.shape}")
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

def f0_to_midi(f0):
    """
    Convert fundamental frequency (F0) values in Hz to MIDI note numbers.
    
    Args:
        f0 (numpy.ndarray): F0 contour in Hz
        
    Returns:
        numpy.ndarray: MIDI note numbers
    """
    # The formula for converting frequency to MIDI note number is:
    # MIDI = 69 + 12 * log2(f0/440)
    # where 69 is A4 (440 Hz)
    
    # Create a copy of the f0 array
    midi_notes = np.zeros_like(f0)
    
    # Only convert non-zero values (voiced regions)
    voiced = f0 > 0
    
    # Apply the formula for voiced regions
    if np.any(voiced):
        midi_notes[voiced] = 69 + 12 * np.log2(f0[voiced]/440)
    
    return midi_notes

def estimate_phoneme_midi_notes(f0, phonemes, hop_length, sample_rate, time_scale=1e-7):
    """
    Estimate the MIDI note for each phoneme based on the F0 contour.
    
    Args:
        f0 (numpy.ndarray): F0 contour
        phonemes (list): List of phoneme tuples (start_time, end_time, phoneme)
        hop_length (int): Hop length used for F0 extraction
        sample_rate (int): Sample rate of the audio
        time_scale (float): Scale factor to convert phoneme timings to seconds
        
    Returns:
        list: List of tuples (phoneme, estimated MIDI note)
    """
    # Convert f0 to MIDI notes
    midi_notes = f0_to_midi(f0)
    
    # Create a list to store the estimated MIDI notes for each phoneme
    phoneme_midi_notes = []
    
    # For each phoneme, calculate the average MIDI note
    for start, end, phone in phonemes:
        # Convert phoneme start and end times to seconds
        start_sec = start * time_scale
        end_sec = end * time_scale
        
        # Convert seconds to frame indices
        start_frame = int(start_sec * sample_rate / hop_length)
        end_frame = int(end_sec * sample_rate / hop_length)
        
        # Ensure frame indices are within the bounds of the f0 array
        start_frame = max(0, start_frame)
        end_frame = min(len(midi_notes), end_frame)
        
        # Extract MIDI notes for this phoneme
        if start_frame < end_frame:
            phoneme_midi = midi_notes[start_frame:end_frame]
            
            # Calculate the average MIDI note for voiced regions
            voiced = phoneme_midi > 0
            if np.any(voiced):
                avg_midi = np.mean(phoneme_midi[voiced])
                # Round to nearest semitone
                midi_note = round(avg_midi)
            else:
                # Unvoiced phoneme (consonant)
                midi_note = 0
        else:
            # Phoneme is too short to have any frames
            midi_note = 0
        
        # Store the phoneme and its MIDI note
        phoneme_midi_notes.append((phone, midi_note))
    
    return phoneme_midi_notes

def midi_to_hz(midi_note):
    """
    Convert a MIDI note number to frequency in Hz.
    
    Args:
        midi_note (float): MIDI note number
        
    Returns:
        float: Frequency in Hz
    """
    if midi_note <= 0:  # Handle zero or negative values
        return 0
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

def hz_to_midi(frequency):
    """
    Convert a frequency in Hz to MIDI note number.
    
    Args:
        frequency (float): Frequency in Hz
        
    Returns:
        float: MIDI note number
    """
    if frequency <= 0:  # Handle zero or negative values
        return 0
    return 69 + 12 * np.log2(frequency / 440.0)

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