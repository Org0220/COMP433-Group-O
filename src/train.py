import torch
import torch.optim as optim
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast

from src.config import (
    SPLITS_DIR,
    IMAGES_DIR,
    LABELS_FILE,
    BATCH_SIZE_BYOL,
    BATCH_SIZE_SUPERVISED,
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    SAVE_PATH
)
from src.datasets import BYOLDataset
from src.transforms import byol_transform
from src.model import BYOL, get_base_encoder
from src.utils import EarlyStopping
from src.utils import set_seed 

def train_byol():

    set_seed(69)

    # Initialize the BYOL model
    base_encoder, feature_dim = get_base_encoder(pretrained=True)
    byol_model = BYOL(base_encoder=base_encoder, feature_dim=feature_dim)
    
    # Move the model to the specified device
    byol_model = byol_model.to(DEVICE)
    
    # Set the model to training mode
    byol_model.train()
    
    # Initialize optimizer
    optimizer = optim.Adam(byol_model.parameters(), lr=LEARNING_RATE)
    
    # Initialize scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=20, verbose=True, delta=0.001)
    
    # Initialize GradScaler for mixed precision
    scaler = GradScaler(device=DEVICE)
    
    # Create BYOL training dataset and DataLoader
    pretrain_split_file = os.path.join(SPLITS_DIR, "train_unlabeled.txt")
    byol_train_dataset = BYOLDataset(
        image_dir=IMAGES_DIR,
        split_file=pretrain_split_file,
        transform=byol_transform
    )
    
    byol_train_loader = torch.utils.data.DataLoader(
        byol_train_dataset,
        batch_size=BATCH_SIZE_BYOL,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        byol_model.train()  # Ensure model is in training mode
        epoch_loss = 0.0
        num_batches = 0

        # Initialize tqdm progress bar
        progress_bar = tqdm(byol_train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)

        for view1, view2 in progress_bar:
            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)

            optimizer.zero_grad()

            with autocast(device_type=DEVICE.type):  # Mixed precision
                loss = byol_model(view1, view2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar with current batch loss
            progress_bar.set_postfix({'Batch Loss': loss.item()})

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Average Loss: {avg_loss:.4f}")

        # Update the scheduler based on the average loss
        scheduler.step(avg_loss)

        # Check early stopping condition
        early_stopping(avg_loss, byol_model, SAVE_PATH)

        if early_stopping.early_stop:
            print("Early stopping triggered. Training halted.")
            break

    # Load the best model before proceeding
    byol_model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    print("Loaded the best model for further training/evaluation.")

    return byol_model

if __name__ == "__main__":
    from transforms import byol_transform
    from data_utils import load_and_verify_data, create_and_save_splits

    # Set seed for reproducibility
    from utils import set_seed
    set_seed(42)

    # Load and verify data
    df, image_filenames, unlabeled_images = load_and_verify_data()
    
    # Create and save splits
    create_and_save_splits(df, image_filenames, unlabeled_images)
    
    # Train BYOL model
    trained_model = train_byol()
