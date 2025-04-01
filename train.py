from data.datasets import test_load_data
from utils.utils_general import load_config

def main():
    """
    Main function for training.
    """
    # Load configuration
    config = load_config()
    
    # Test loading data
    print("Testing data loading...")
    test_load_data()
    
    # Rest of the training code would go here
    print("Ready for training.")

if __name__ == "__main__":
    main()