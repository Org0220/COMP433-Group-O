import os
import json
from src.utils import set_seed
from src.data_utils import load_and_verify_data, create_and_save_splits
from src.train import train_byol
from src.config import (
    SPLITS_DIR,
    IMAGES_DIR,
    LABELS_FILE,
    BATCH_SIZE_BYOL,
    BATCH_SIZE_SUPERVISED,
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE
)

def main():
    # Prompt the user for a run name
    run_name = input("Enter a name for this training run: ").strip()
    
    if not run_name:
        print("Run name cannot be empty. Please try again.")
        return
    
    # Define the runs directory
    runs_dir = os.path.join(os.getcwd(), 'runs')
    os.makedirs(runs_dir, exist_ok=True)
    
    # Define the specific run directory
    run_dir = os.path.join(runs_dir, run_name)
    
    # Determine if resuming or starting a new run
    if os.path.exists(run_dir):
        print(f"A run with the name '{run_name}' already exists.")
        resume_input = input("Do you want to resume this run? (y/n): ").strip().lower()
        if resume_input == 'y':
            resume = True
            print(f"Resuming run '{run_name}'...")
        else:
            print("Please choose a different run name to start a new run.")
            return
    else:
        os.makedirs(run_dir)
        print(f"Created run directory at: {run_dir}")
        resume = False
    
    # Set seed for reproducibility
    set_seed()
    
    # Load and verify data
    df, image_filenames, unlabeled_images = load_and_verify_data()
    
    # Create and save splits
    create_and_save_splits(df, image_filenames, unlabeled_images)
    
    # Save configuration to run_dir
    config = {
        "run_name": run_name,
        "batch_size_byol": BATCH_SIZE_BYOL,
        "batch_size_supervised": BATCH_SIZE_SUPERVISED,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau",
        "patience": 10,
        "delta": 0.001,
        "device": str(DEVICE),
        "split_ratios": {
            "train": 0.7,
            "val": 0.15,
            "test": 0.15
        }
        # Add more configurations as needed
    }
    
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to: {config_path}")
    
    # Train BYOL model
    trained_model = train_byol(run_dir, resume=resume)
    
    # Further steps like evaluation or supervised training can be added here

if __name__ == "__main__":
    main()
