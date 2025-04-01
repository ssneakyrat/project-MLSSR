import h5py
import os
import random
from utils.utils_general import load_config

def test_load_data():
    """
    Test function to load data from h5 binary files and print 3 random sample file names.
    
    Returns:
        list: List of file names for the 3 random samples
    """
    # Load configuration
    config = load_config()
    
    # Get the path to the binary directory and file
    bin_dir = config['data']['bin_dir']
    bin_file = config['data']['bin_file']
    bin_path = os.path.join(bin_dir, bin_file)
    
    print(f"Loading data from {bin_path}")
    
    # Check if the binary file exists
    if not os.path.exists(bin_path):
        print(f"Error: Binary file {bin_path} not found!")
        return []
    
    try:
        with h5py.File(bin_path, 'r') as f:
            # Check if there's a data group
            if 'data' in f:
                data_group = f['data']
                
                # Check if FILE_NAME exists in the data group
                if 'FILE_NAME' in data_group:
                    file_names = data_group['FILE_NAME'][:]
                    
                    # Select 3 random indices or fewer if there are less than 3 samples
                    num_samples = min(3, len(file_names))
                    random_indices = random.sample(range(len(file_names)), num_samples)
                    
                    # Get the file names for the selected indices
                    selected_files = [file_names[i] for i in random_indices]
                    
                    # Print the selected file names
                    print(f"Selected {num_samples} random samples:")
                    for i, file_name in enumerate(selected_files):
                        print(f"  {i+1}. {file_name}")
                    
                    return selected_files
                else:
                    print("Error: FILE_NAME not found in the data group")
            else:
                print("Error: 'data' group not found in the binary file")
                
            # If we couldn't find the expected structure, look for individual files
            print("Looking for individual h5 files...")
            
    except Exception as e:
        print(f"Error loading data from binary file: {e}")
    
    # Alternative approach: Find individual h5 files in the bin_dir
    try:
        # Get all h5 files in the bin_dir
        h5_files = [f for f in os.listdir(bin_dir) if f.endswith('.h5')]
        
        if not h5_files:
            print(f"No h5 files found in {bin_dir}")
            return []
        
        # Select 3 random files
        num_samples = min(3, len(h5_files))
        selected_h5_files = random.sample(h5_files, num_samples)
        
        # Read the file names from the selected h5 files
        selected_files = []
        
        print(f"Selected {num_samples} random h5 files:")
        for i, h5_file in enumerate(selected_h5_files):
            file_path = os.path.join(bin_dir, h5_file)
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'data' in f and 'FILE_NAME' in f['data']:
                        file_name = f['data']['FILE_NAME'][0]
                        selected_files.append(file_name)
                        print(f"  {i+1}. {h5_file}: {file_name}")
                    else:
                        print(f"  {i+1}. {h5_file}: Warning - FILE_NAME not found")
            except Exception as e:
                print(f"  {i+1}. {h5_file}: Error - {e}")
        
        return selected_files
        
    except Exception as e:
        print(f"Error finding individual h5 files: {e}")
        return []