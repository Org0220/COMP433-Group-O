import random
import numpy as np
import torch

def set_seed(seed=69):
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    Args:
        seed (int): The seed value to use. Default is 42.
    """
    random.seed(seed)                     # Python random module
    np.random.seed(seed)                  # NumPy
    torch.manual_seed(seed)               # PyTorch CPU
    torch.cuda.manual_seed(seed)          # PyTorch GPU
    torch.cuda.manual_seed_all(seed)      # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    """
    Early stops the training if the loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
            verbose (bool): If True, prints a message for each loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_path = None

    def __call__(self, loss, model, save_path='best_byol_model.pth'):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model, save_path)
        elif loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = loss
            self.save_checkpoint(model, save_path)
            self.counter = 0

    def save_checkpoint(self, model, save_path):
        """Saves model when loss decreases."""
        torch.save(model.state_dict(), save_path)
        if self.verbose:
            print(f'Validation loss decreased. Saving model to {save_path}')
