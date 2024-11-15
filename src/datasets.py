import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class BYOLDataset(Dataset):
    def __init__(self, image_dir, split_file, transform=None):
        self.image_dir = image_dir
        with open(split_file, 'r') as f:
            self.image_files = f.read().splitlines()
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            # Generate two augmented views
            view1 = self.transform(image)
            view2 = self.transform(image)
        return view1, view2

class LabeledDataset(Dataset):
    def __init__(self, image_dir, labels_csv, split_file, transform=None):
        self.image_dir = image_dir
        self.labels_df = pd.read_csv(labels_csv)
        with open(split_file, "r") as f:
            self.image_files = f.read().splitlines()
        self.labels_df = self.labels_df[
            self.labels_df["Image"].isin(self.image_files)
        ].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.labels_df.loc[self.labels_df["Image"] == img_name, "Class"].values[0]
        if self.transform:
            image = self.transform(image)
        return image, label
