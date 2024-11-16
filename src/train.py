import torch
import torch.optim as optim
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

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
from src.datasets import BYOLDataset
from src.model import BYOL, get_base_encoder
from src.utils import EarlyStopping
from src.transforms import byol_transform
from src.utils import set_seed


def worker_init_fn(worker_id):
    """
    Initializes each DataLoader worker with a different seed.

    Args:
        worker_id (int): Worker identifier.
    """
    # Retrieve the base seed
    base_seed = torch.initial_seed()
    # Set the seed for each worker
    np.random.seed(base_seed % 2**32 + worker_id)
    random.seed(base_seed % 2**32 + worker_id)


def train_byol(run_dir, resume=False):
    """
    Trains the BYOL model and saves checkpoints and TensorBoard logs.
    Can resume training from a saved checkpoint if resume=True.

    Args:
        run_dir (str): Directory path where the run's artifacts will be saved.
        resume (bool): Whether to resume training from the last checkpoint.
    """
    # Initialize TensorBoard SummaryWriter
    tb_log_dir = os.path.join(run_dir, "tensorboard_logs")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs will be saved to: {tb_log_dir}")

    # Initialize the BYOL model
    base_encoder, feature_dim = get_base_encoder(pretrained=True)
    byol_model = BYOL(base_encoder=base_encoder, feature_dim=feature_dim)

    # Move the model to the specified device
    byol_model = byol_model.to(DEVICE)

    # Set the model to training mode
    byol_model.train()

    # Initialize optimizer
    optimizer = optim.Adam(byol_model.parameters(), lr=LEARNING_RATE)

    # Initialize scheduler without verbose to avoid warning
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Initialize EarlyStopping with run_dir for saving the best model
    early_stopping = EarlyStopping(
        patience=15, verbose=True, delta=0.001, run_dir=run_dir
    )

    # Initialize GradScaler for mixed precision with device specified
    scaler = GradScaler(device=DEVICE)

    # Create BYOL training dataset and DataLoader
    pretrain_split_file = os.path.join(SPLITS_DIR, "train_unlabeled.txt")
    byol_train_dataset = BYOLDataset(
        image_dir=IMAGES_DIR, split_file=pretrain_split_file, transform=byol_transform
    )

    byol_train_loader = torch.utils.data.DataLoader(
        byol_train_dataset,
        batch_size=BATCH_SIZE_BYOL,
        shuffle=True,
        num_workers=4,  # Adjust based on your system
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    start_epoch = 1  # Default start epoch

    if resume:
        # Load checkpoint
        early_stopping.load_checkpoint(byol_model, optimizer, scheduler, scaler, map_location=DEVICE)
        # Retrieve last epoch from checkpoint
        try:
            checkpoint = torch.load(early_stopping.best_model_path, map_location=DEVICE)
            start_epoch = checkpoint.get('epoch', 1) + 1
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"Failed to retrieve epoch from checkpoint. Error: {e}. Starting from epoch 1.")
            start_epoch = 1

    # Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        byol_model.train()  # Ensure model is in training mode
        epoch_loss = 0.0
        num_batches = 0

        # Initialize tqdm progress bar
        progress_bar = tqdm(
            byol_train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False
        )

        for view1, view2 in progress_bar:
            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)

            optimizer.zero_grad()

            with autocast(device_type=DEVICE.type):
                loss = byol_model(view1, view2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar with current batch loss
            progress_bar.set_postfix({"Batch Loss": loss.item()})

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Average Loss: {avg_loss:.4f}")

        # Log average loss to TensorBoard
        writer.add_scalar("Loss/Average", avg_loss, epoch)

        # Update the scheduler based on the average loss
        scheduler.step(avg_loss)

        # Check early stopping condition
        early_stopping(avg_loss, byol_model, optimizer, scheduler, scaler, epoch)

        # Log Learning Rate to TensorBoard
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning Rate", current_lr, epoch)

        if early_stopping.early_stop:
            print("Early stopping triggered. Training halted.")
            break

    # Close the TensorBoard writer
    writer.close()

    # Load the best model before proceeding with weights_only=True
    # Avoid loading the entire checkpoint into the model
    try:
        checkpoint = torch.load(early_stopping.best_model_path, map_location=DEVICE)
        byol_model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded the best model for further training/evaluation.")
    except KeyError as e:
        print(f"KeyError during model loading: {e}")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

    return byol_model
