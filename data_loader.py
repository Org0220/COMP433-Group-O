import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class LabeledDataset(Dataset):
    """Dataset for supervised learning with labeled images."""
    def __init__(self, excel_path, transform=None):
        self.data = pd.read_excel(excel_path)  # Load Excel
        self.image_paths = self.data['image_path'].values  # Image paths
        self.labels = self.data['label'].values  # Labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")  # Load image

        if self.transform:
            image = self.transform(image)

        return image, label