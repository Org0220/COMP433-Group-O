import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np
import plotly.express as px
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import (
    SPLITS_DIR,
    IMAGES_DIR,
    LABELS_FILE,
    BATCH_SIZE_BYOL,
    BATCH_SIZE_SUPERVISED,
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    LEARNING_RATE_FROZEN,
    LEARNING_RATE_UNFROZEN,
)
from src.datasets import BYOLDataset, LabeledDataset
from src.model import BYOL, SupervisedModel, get_base_encoder
from src.utils import EarlyStopping, WorkerInitializer
from src.transforms import byol_transform
from src.utils import set_seed


def get_worker_init_fn(seed):
    """
    Returns a worker initialization function with a fixed seed.

    Args:
        seed (int): Base seed for worker initialization
    """

    def worker_init_fn_fixed(worker_id):
        worker_seed = (seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn_fixed


class BYOLDataLoader(torch.utils.data.DataLoader):
    def set_epoch(self, epoch):
        self.dataset.epoch = epoch


def train_byol(run_dir, resume=False, custom_resnet=False):
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

    # Initialize the BYOL model
    base_encoder, feature_dim = get_base_encoder(pretrained=True, custom_resnet=custom_resnet)
    byol_model = BYOL(base_encoder=base_encoder, feature_dim=feature_dim)

    # Move the model to the specified device
    byol_model = byol_model.to(DEVICE)

    # Set the model to training mode
    byol_model.train()

    # Initialize optimizer
    optimizer = optim.Adam(byol_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Initialize scheduler without verbose to avoid warning
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Initialize EarlyStopping with run_dir for saving the best model
    early_stopping = EarlyStopping(
        patience=18, verbose=True, delta=0.001, run_dir=run_dir
    )

    # Initialize GradScaler for mixed precision with device specified
    scaler = GradScaler(device=DEVICE)

    # Create BYOL training dataset and DataLoader
    pretrain_split_file = os.path.join(SPLITS_DIR, "train_unlabeled.txt")
    byol_train_dataset = BYOLDataset(
        image_dir=IMAGES_DIR, split_file=pretrain_split_file, transform=byol_transform
    )

    # Create deterministic generator with fixed seed
    g = torch.Generator()
    g.manual_seed(torch.initial_seed())

    # Create worker initializer with fixed seed
    worker_init = WorkerInitializer(base_seed=torch.initial_seed())

    byol_train_loader = BYOLDataLoader(
        byol_train_dataset,
        batch_size=BATCH_SIZE_BYOL,
        shuffle=True,
        num_workers=4,  # Ensure this is set as per reproducibility needs
        pin_memory=True,
        worker_init_fn=worker_init,  # Use WorkerInitializer instance
        generator=g,
        persistent_workers=True,
    )

    start_epoch = 1  # Default start epoch

    if resume:
        # Load checkpoint
        early_stopping.load_checkpoint(
            byol_model, optimizer, scheduler, scaler, map_location=DEVICE
        )
        # Retrieve last epoch from checkpoint
        try:
            checkpoint = torch.load(
                early_stopping.best_model_path, map_location=DEVICE, weights_only=True
            )
            start_epoch = checkpoint.get("epoch", 1) + 1
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(
                f"Failed to retrieve epoch from checkpoint. Error: {e}. Starting from epoch 1."
            )
            start_epoch = 1

    # Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        byol_train_loader.set_epoch(epoch)  # Update epoch for new augmentations
        print(f"Starting Epoch {epoch}")
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


        print(f"Epoch {epoch} completed.")

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] - Average Loss: {avg_loss:.4f}")

        # Log average loss to TensorBoard
        writer.add_scalar("Loss/Average", avg_loss, epoch)

        # Update the scheduler based on the average loss
        scheduler.step(avg_loss)
        print(f"Scheduler step completed for epoch {epoch}.")

        # Check early stopping condition
        early_stopping(avg_loss, byol_model, optimizer, scheduler, scaler, epoch)

        # Log Learning Rate to TensorBoard
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Learning Rate", current_lr, epoch)
        print(f"Current Learning Rate: {current_lr}")

        if early_stopping.early_stop:
            print("Early stopping triggered. Training halted.")
            break

    # Close the TensorBoard writer
    writer.close()
    print("TensorBoard writer closed.")

    # Load the best model before proceeding with weights_only=True
    # Avoid loading the entire checkpoint into the model
    try:
        checkpoint = torch.load(
            early_stopping.best_model_path, map_location=DEVICE, weights_only=True
        )
        byol_model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded the best model for further training/evaluation.")
    except KeyError as e:
        print(f"KeyError during model loading: {e}")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

    return byol_model


def evaluate_model(model, test_loader, device):
    """
    Evaluates the model on the test set.

    Args:
        model (nn.Module): The trained model to evaluate
        test_loader (DataLoader): DataLoader for the test set
        device (torch.device): Device to run evaluation on

    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # For per-class accuracy
    class_correct = {}
    class_total = {}

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating on test set", leave=True):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for metric calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate per-class accuracy
            for label, pred in zip(labels, predicted):
                label_item = label.item()
                if label_item not in class_correct:
                    class_correct[label_item] = 0
                    class_total[label_item] = 0
                class_total[label_item] += 1
                if label_item == pred.item():
                    class_correct[label_item] += 1

    # Calculate metrics
    test_accuracy = 100 * correct / total
    avg_test_loss = test_loss / len(test_loader)
    per_class_accuracy = {
        class_idx: 100 * class_correct[class_idx] / class_total[class_idx]
        for class_idx in class_correct.keys()
    }

    # Calculate additional metrics
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Plot confusion matrix
    fig_cm = px.imshow(cm,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[f"Class {i}" for i in range(cm.shape[1])],
                    y=[f"Class {i}" for i in range(cm.shape[0])])

    fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label", yaxis_title="True Label")
    fig_cm.show()

    return {
        'accuracy': test_accuracy,
        'loss': avg_test_loss,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'per_class_accuracy': per_class_accuracy
    }


def train_supervised(run_dir, resume=False, num_classes=7, pretrained=True, freeze_layers=None):
    """
    Trains the model in supervised fashion and saves checkpoints and TensorBoard logs.

    Args:
        run_dir (str): Directory path where the run's artifacts will be saved.
        resume (bool): Whether to resume training from the last checkpoint.
        num_classes (int): Number of target classes.
        pretrained (bool): Whether to use ImageNet pretrained weights if BYOL weights fail to load.
        freeze_layers (str, optional): Which layers to freeze ('all', 'partial', or None)
    """
    # Initialize TensorBoard SummaryWriter
    tb_log_dir = os.path.join(run_dir, "tensorboard_logs_supervised")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs will be saved to: {tb_log_dir}")

    # Initialize the base encoder
    base_encoder, feature_dim = get_base_encoder(
        pretrained=False
    )  # Initially no pretrained weights

    # Load the pretrained BYOL encoder
    checkpoint_path = os.path.join(run_dir, "best_byol_model.pth")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            # First, get the online encoder's state dict
            byol_state_dict = checkpoint["model_state_dict"]

            # Create a new state dict for the base encoder
            encoder_state_dict = {}

            # Map BYOL keys to ResNet keys
            for key, value in byol_state_dict.items():
                if key.startswith("online_encoder."):
                    # Remove 'online_encoder.' prefix and map directly to ResNet keys
                    new_key = key.replace("online_encoder.", "")
                    if new_key in base_encoder.state_dict():
                        encoder_state_dict[new_key] = value

            # Check if we found any matching keys
            if len(encoder_state_dict) == 0:
                raise ValueError("No matching keys found in BYOL checkpoint")

            # Load the weights
            base_encoder.load_state_dict(encoder_state_dict, strict=False)
            print(
                f"Successfully loaded {len(encoder_state_dict)} layers from BYOL checkpoint."
            )

        except Exception as e:
            print(f"Error loading BYOL weights: {e}")
            # If BYOL weights fail to load and pretrained=True, reinitialize with ImageNet weights
            if pretrained:
                print("Falling back to ImageNet pretrained weights...")
                base_encoder, feature_dim = get_base_encoder(pretrained=True)
            else:
                print("Using random initialization...")
    else:
        print(f"No BYOL checkpoint found at {checkpoint_path}.")
        if pretrained:
            print("Using ImageNet pretrained weights...")
            base_encoder, feature_dim = get_base_encoder(pretrained=True)
        else:
            print("Using random initialization...")

    # Initialize the supervised model with the loaded encoder and freezing option
    supervised_model = SupervisedModel(
        base_encoder, 
        feature_dim, 
        num_classes,
        freeze_layers=freeze_layers
    ).to(DEVICE)

    # Choose learning rate based on freezing strategy
    if freeze_layers in ['all', 'partial']:
        learning_rate = LEARNING_RATE_FROZEN
        print(f"Using frozen learning rate: {LEARNING_RATE_FROZEN}")
    else:
        learning_rate = LEARNING_RATE_UNFROZEN
        print(f"Using unfrozen learning rate: {LEARNING_RATE_UNFROZEN}")

    # Initialize optimizer with appropriate learning rate
    optimizer = optim.Adam(
        supervised_model.parameters(), 
        lr=learning_rate,  # Use the selected learning rate
        weight_decay=1e-4
    )

    # Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(
        patience=10, verbose=True, delta=0.001, run_dir=run_dir
    )

    # Initialize GradScaler for mixed precision
    scaler = GradScaler(device=DEVICE)

    # Create supervised training dataset and DataLoader using LabeledDataset
    train_split_file = os.path.join(SPLITS_DIR, "train_labeled.txt")
    val_split_file = os.path.join(SPLITS_DIR, "val_labeled.txt")

    supervised_train_dataset = LabeledDataset(
        image_dir=IMAGES_DIR,
        labels_csv=LABELS_FILE,
        split_file=train_split_file,
        transform=byol_transform,
    )

    supervised_val_dataset = LabeledDataset(
        image_dir=IMAGES_DIR,
        labels_csv=LABELS_FILE,
        split_file=val_split_file,
        transform=byol_transform,
    )

    # Create worker initializer with fixed seed
    worker_init = WorkerInitializer(base_seed=torch.initial_seed())

    supervised_train_loader = torch.utils.data.DataLoader(
        supervised_train_dataset,
        batch_size=BATCH_SIZE_SUPERVISED,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init,  # Use WorkerInitializer instance
    )

    supervised_val_loader = torch.utils.data.DataLoader(
        supervised_val_dataset,
        batch_size=BATCH_SIZE_SUPERVISED,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=worker_init,  # Use WorkerInitializer instance
    )

    start_epoch = 1  # Default start epoch

    if resume:
        # Load supervised checkpoint if exists
        checkpoint_supervised = os.path.join(run_dir, "best_supervised_model.pth")
        if os.path.exists(checkpoint_supervised):
            checkpoint = torch.load(checkpoint_supervised, map_location=DEVICE)
            supervised_model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint.get("epoch", 1) + 1
            early_stopping.best_loss = checkpoint.get("best_loss", None)
            early_stopping.counter = checkpoint.get("early_stopping_counter", 0)
            print(f"Resuming supervised training from epoch {start_epoch}")
        else:
            print(
                f"No supervised checkpoint found at {checkpoint_supervised}. Starting fresh."
            )

    # Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        supervised_model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0
        num_batches = 0

        progress_bar = tqdm(
            supervised_train_loader,
            desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]",
            leave=True,
        )

        for inputs, labels in progress_bar:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            if len(labels.shape) > 1:
                labels = labels.squeeze()

            optimizer.zero_grad()

            with autocast(device_type=DEVICE.type):
                outputs = supervised_model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            epoch_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / num_batches
        train_accuracy = 100 * train_correct / train_total

        # Validation
        supervised_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar_val = tqdm(
                supervised_val_loader,
                desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]",
                leave=False,
            )
            for inputs, labels in progress_bar_val:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                with autocast(device_type=DEVICE.type):
                    outputs = supervised_model(inputs)
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(supervised_val_loader)
        val_accuracy = 100 * correct / total

        # Print epoch summary with both accuracies
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Train Acc: {train_accuracy:.2f}%, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.2f}%"
        )

        # Log to tensorboard
        writer.add_scalar("Train/Loss", avg_train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_accuracy, epoch)
        writer.add_scalar("Val/Loss", avg_val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_accuracy, epoch)

        # Update the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Early stopping check
        early_stopping(
            avg_val_loss, supervised_model, optimizer, scheduler, scaler, epoch
        )
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Close the TensorBoard writer
    writer.close()

    # Load the best model before evaluation
    try:
        checkpoint = torch.load(early_stopping.best_model_path, map_location=DEVICE)
        supervised_model.load_state_dict(checkpoint["model_state_dict"])
        print("Loaded the best supervised model for evaluation.")
    except Exception as e:
        print(f"Error loading best model: {e}")
        print("Using current model state for evaluation.")

    # Create test dataset and loader
    test_split_file = os.path.join(SPLITS_DIR, "test.txt")
    test_dataset = LabeledDataset(
        image_dir=IMAGES_DIR,
        labels_csv=LABELS_FILE,
        split_file=test_split_file,
        transform=byol_transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE_SUPERVISED,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Evaluate on test set
    test_metrics = evaluate_model(supervised_model, test_loader, DEVICE)

    # Print test results
    print("\nTest Set Evaluation Results:")
    print(f"Overall Test Accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"Overall Test Loss: {test_metrics['loss']:.4f}")
    print(f"Overall Test F1 Score: {test_metrics['f1_score']:.4f}")
    print(f"Overall Test Precision: {test_metrics['precision']:.4f}")
    print(f"Overall Test Recall: {test_metrics['recall']:.4f}")
    print("\nPer-class Test Accuracies:")

    # Get class names from the dataset
    class_names = {v: k for k, v in test_dataset.class_to_idx.items()}
    for class_idx, accuracy in test_metrics['per_class_accuracy'].items():
        class_name = class_names[class_idx]
        print(f"{class_name}: {accuracy:.2f}%")

    # Save test results to a file
    results_file = os.path.join(run_dir, "test_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.2f}%\n")
        f.write(f"Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"Test F1 Score: {test_metrics['f1_score']:.4f}\n")
        f.write(f"Test Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Test Recall: {test_metrics['recall']:.4f}\n\n")
        f.write("Per-class Test Accuracies:\n")
        for class_idx, accuracy in test_metrics['per_class_accuracy'].items():
            class_name = class_names[class_idx]
            f.write(f"{class_name}: {accuracy:.2f}%\n")

    # Log final test metrics to tensorboard
    writer.add_scalar("Test/Accuracy", test_metrics['accuracy'], 0)
    writer.add_scalar("Test/Loss", test_metrics['loss'], 0)
    writer.add_scalar("Test/F1_Score", test_metrics['f1_score'], 0)
    writer.add_scalar("Test/Precision", test_metrics['precision'], 0)
    writer.add_scalar("Test/Recall", test_metrics['recall'], 0)
    for class_idx, accuracy in test_metrics['per_class_accuracy'].items():
        writer.add_scalar(f"Test/Class_{class_idx}_Accuracy", accuracy, 0)

    return supervised_model
