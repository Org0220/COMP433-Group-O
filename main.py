from src.utils import set_seed
from src.data_utils import load_and_verify_data, create_and_save_splits
from src.train import train_byol

def main():
    
    # Set seed for reproducibility
    set_seed(69)
    
    # Load and verify data
    df, image_filenames, unlabeled_images = load_and_verify_data()
    
    # Create and save splits
    create_and_save_splits(df, image_filenames, unlabeled_images)
    
    # Train BYOL model
    trained_model = train_byol()
    
    # Further steps like evaluation or supervised training can be added here

if __name__ == "__main__":
    main()
