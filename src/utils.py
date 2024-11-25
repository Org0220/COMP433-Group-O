import random
import numpy as np
import torch
import os


def set_seed(seed, deterministic=True):
    """
    Sets all seeds for reproducibility.

    Args:
        seed (int): The seed value to use
        deterministic (bool): Whether to enforce deterministic behavior
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            print(f"Warning: Unable to enable deterministic mode: {e}")
            print("Training will continue with non-deterministic behavior")


def get_worker_seed(worker_id, base_seed):
    """
    Generate a deterministic seed for each worker.

    Args:
        worker_id (int): ID of the worker
        base_seed (int): Base seed to derive worker seed from

    Returns:
        int: Seed for the worker
    """
    return base_seed + worker_id


class WorkerInitializer:
    """
    Class to handle worker initialization with proper pickling support.
    """

    def __init__(self, base_seed):
        self.base_seed = base_seed

    def __call__(self, worker_id):
        worker_seed = (self.base_seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


class EarlyStopping:
    """
    Early stops the training if the loss doesn't improve after a given patience.
    Additionally, manages saving and loading of training state.
    """

    def __init__(self, patience=5, verbose=False, delta=0, run_dir=None, mode="byol"):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
            verbose (bool): If True, prints a message for each loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            run_dir (str): Directory path where the best model is saved.
            mode (str): 'byol' or 'supervised' to differentiate checkpoint filenames.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.run_dir = run_dir
        if mode == "byol":
            self.best_model_path = (
                os.path.join(run_dir, "best_byol_model.pth")
                if run_dir
                else "best_byol_model.pth"
            )
        elif mode == "supervised":
            self.best_model_path = (
                os.path.join(run_dir, "best_supervised_model.pth")
                if run_dir
                else "best_supervised_model.pth"
            )
        else:
            raise ValueError("Mode should be either 'byol' or 'supervised'")

    def __call__(self, loss, model, optimizer, scheduler, scaler, epoch):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model, optimizer, scheduler, scaler, epoch)
        elif loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.save_checkpoint(model, optimizer, scheduler, scaler, epoch)
            self.counter = 0

    def save_checkpoint(self, model, optimizer, scheduler, scaler, epoch):
        """Saves model and training state when loss decreases."""
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "best_loss": self.best_loss,
            "early_stopping_counter": self.counter,
        }
        torch.save(checkpoint, self.best_model_path)
        if self.verbose:
            print(f"Validation loss decreased. Saving model to {self.best_model_path}")

    def load_checkpoint(self, model, optimizer, scheduler, scaler, map_location):
        """Loads model and training state from the checkpoint."""
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=map_location)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            self.best_loss = checkpoint["best_loss"]
            self.counter = checkpoint["early_stopping_counter"]
            print(
                f"Loaded checkpoint from {self.best_model_path} at epoch {checkpoint['epoch']} with best loss {self.best_loss:.4f}"
            )
        else:
            print(f"No checkpoint found at {self.best_model_path}. Starting fresh.")
