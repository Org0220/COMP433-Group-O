import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import hashlib
from src.transforms import BYOLTransform


class BYOLDataset(Dataset):
    def __init__(self, image_dir, split_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.epoch = 0  # Add epoch tracking

        # Read the split file
        with open(split_file, "r") as f:
            lines = f.readlines()

        # Parse the lines into image paths and labels
        self.samples = []
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) == 2:  # If there's a label
                img_path, label = parts
                self.samples.append((img_path, int(label)))  # Convert label to int
            else:  # If no label (for unlabeled data)
                img_path = parts[0]
                self.samples.append((img_path, None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_path)
        image = Image.open(img_path).convert("RGB")

        if label is not None:  # Supervised case
            if self.transform:
                torch.manual_seed(torch.initial_seed() + idx)
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.long)

        # For BYOL, create two different views
        if self.transform:
            view1, view2 = self.transform(image)  # The transform should be an instance of BYOLTransform
            return view1, view2

        return image, image


class LabeledDataset(Dataset):
    def __init__(self, image_dir, labels_csv, split_file, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_csv)
        with open(split_file, "r") as f:
            self.image_files = [
                line.strip().split(",")[0] for line in f.readlines()
            ]  # Get just the image paths

        # Create a mapping of class names to indices
        self.class_to_idx = {
            name: idx
            for idx, name in enumerate(sorted(self.labels_df["Class"].unique()))
        }

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Set seed based on index for reproducible augmentations
        seed = torch.initial_seed() + idx
        torch.manual_seed(seed)

        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Get class name and convert to index
        class_name = self.labels_df.loc[
            self.labels_df["Image"] == img_name, "Class"
        ].values[0]
        label = self.class_to_idx[class_name]

        if self.transform:
            # Check if transform is BYOLTransform
            if isinstance(self.transform, BYOLTransform):
                image, _ = self.transform(image)  # Only use first view if it's BYOLTransform
            else:
                image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
