import os
import glob

def list_dataset_files():
    """
    List all files in the datasets directory.
    """
    # Check if datasets directory exists
    if not os.path.exists("datasets"):
        print("Error: datasets directory not found!")
        return
    
    # Get all files in the datasets directory and its subdirectories
    files = glob.glob("datasets/**/*", recursive=True)
    
    # Filter out directories, keep only files
    files = [f for f in files if os.path.isfile(f)]
    
    # Print the list of files
    print(f"Found {len(files)} files in datasets directory:")
    for file in files:
        print(f"  - {file}")

if __name__ == "__main__":
    list_dataset_files()