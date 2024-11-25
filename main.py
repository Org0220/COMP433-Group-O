import os
import json
from src.utils import set_seed
from src.data_utils import load_and_verify_data, create_and_save_splits
from src.train import train_byol, train_supervised
from src.config import (
    SPLITS_DIR,
    IMAGES_DIR,
    LABELS_FILE,
    BATCH_SIZE_BYOL,
    BATCH_SIZE_SUPERVISED,
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
)


def main():
    # Prompt the user for a run name
    run_name = input("Enter a name for this training run: ").strip()

    if not run_name:
        print("Run name cannot be empty. Please try again.")
        return

    # Set seed before any other operations
    curr_seed = 2
    print(f"Setting seed to: {curr_seed}")
    set_seed(curr_seed, deterministic=True)

    # Define the runs directory
    runs_dir = os.path.join(os.getcwd(), "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # Define the specific run directory
    run_dir = os.path.join(runs_dir, run_name)

    # Initialize mode_input as None
    mode_input = None

    # Determine if resuming or starting a new run
    if os.path.exists(run_dir):
        print(f"A run with the name '{run_name}' already exists.")
        # Ask user to choose training mode upon resuming
        mode_input = input(
            "Choose training mode: [1] Continue BYOL, [2] Supervised Training: "
        ).strip()
        if mode_input == "1":
            resume_byol = True
            resume_supervised = False
            print(f"Resuming BYOL training for run '{run_name}'...")
        elif mode_input == "2":
            resume_byol = False
            resume_supervised = True
            print(f"Starting Supervised Training for run '{run_name}'...")
        else:
            print("Invalid input. Please choose either 1 or 2.")
            return
    else:
        os.makedirs(run_dir)
        # Default to starting BYOL training for new runs
        resume_byol = False
        resume_supervised = False

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
        "scheduler": "CosineAnnealingLR",
        "patience_byol": 18,
        "patience_supervised": 10,
        "delta": 0.001,
        "device": str(DEVICE),
        "split_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "training_mode": "BYOL" if not os.path.exists(run_dir) else "Resumed",
        # Add more configurations as needed
    }

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Train the appropriate model based on run conditions
    if os.path.exists(run_dir) and mode_input:  # Check both conditions
        if mode_input == "1":
            # Continue BYOL training
            trained_model = train_byol(run_dir, resume=True)
        elif mode_input == "2":
            # Start Supervised Training
            trained_model = train_supervised(
                run_dir, 
                resume=True, 
                num_classes=7,
                pretrained=True
            )
    else:
        # Start BYOL training for new run
        trained_model = train_byol(run_dir, resume=False)

    return trained_model


if __name__ == "__main__":
    main()
