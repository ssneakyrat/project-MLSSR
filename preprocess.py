import os
import glob


from utils.utils_general import load_config

def list_lab_files(raw_dir="raw_dir/lab"):
    """
    List all .lab files in the specified directory.
    """
    # Check if directory exists
    if not os.path.exists(raw_dir):
        print(f"Error: {raw_dir} directory not found!")
        return []
    
    # Get all .lab files in the directory
    files = glob.glob(f"{raw_dir}/**/*.lab", recursive=True)
    
    # Print the list of files
    print(f"Found {len(files)} .lab files in {raw_dir} directory:")
    for file in files:
        print(f"  - {file}")
    
    return files

def parse_lab_file(file_path):
    """
    Parse a .lab file containing phoneme timing data.
    """
    phonemes = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Split line into components
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time = int(parts[0])
                    end_time = int(parts[1])
                    phoneme = parts[2]
                    phonemes.append((start_time, end_time, phoneme))
    except Exception as e:
        print(f"Error parsing file {file_path}: {e}")
    
    return phonemes

def main():
    """
    Main function to process lab files.
    """
    # Load configuration
    config = load_config()
    
    # Get list of .lab files
    file_path = config['data']['raw_dir']
    lab_files = list_lab_files(file_path)
    
    # Process each file
    for file_path in lab_files:
        print(f"\nProcessing file: {file_path}")
        phonemes = parse_lab_file(file_path)
        
        # Print phoneme data
        print(f"Found {len(phonemes)} phonemes:")
        for start, end, phoneme in phonemes:
            print(f"  {start}, {end}, {phoneme}")

if __name__ == "__main__":
    main()