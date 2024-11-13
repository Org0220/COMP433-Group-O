import os
import pandas as pd

# Creating file paths for data to be read
data_dir =  os.getcwd()
images_dir = os.path.join(data_dir, 'Sliced_Images')
labels_file = os.path.join(data_dir, 'image_class_mapping.xlsx')


# Loading excel labels with pandas and checking if classes are read properly
df = pd.read_excel(labels_file)

required_columns = {'Image', 'Class'}
if not required_columns.issubset(df.columns):
    missing = required_columns - set(df.columns)
    raise ValueError(f"Excel file is missing the following required columns: {missing}")

# Check how many total images and total labels
image_filenames = set(os.listdir(images_dir))
excel_filenames = set(df['Image'])
print(len(excel_filenames))
print(len(image_filenames))

# Check if all excel labeled data correspond to an image
missing_images = excel_filenames - image_filenames
if missing_images:
    print("Warning: The following images listed in the Excel file are missing in the Sliced_Images directory:")
    for img in missing_images:
        print(f" - {img}")
    df = df[df['Image'].isin(image_filenames)]
    print(f"Filtered DataFrame to {len(df)} labeled images.")
else:
    print("All labeled images are present in the Sliced_Images directory.")

# Print total number of unlabeled data
unlabeled_images = image_filenames - excel_filenames
print(f"Total unlabeled images: {len(unlabeled_images)}")

# Print total number of unique classes
classes = sorted(df['Class'].unique())
print(f"Found {len(classes)} unique classes: {classes}")


from sklearn.model_selection import train_test_split

splits = ['train', 'val', 'test']

# Creating file paths for test/train/val with labels
for split in splits:
    for class_name in classes:
        split_class_dir = os.path.join(data_dir, split, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

df['filepath'] = df['Image'].apply(lambda x: os.path.join(images_dir, x))

# Verify that all file paths exist
missing_files = df[~df['filepath'].apply(os.path.exists)]
if not missing_files.empty:
    print("Warning: The following file paths do not exist and will be excluded:")
    for _, row in missing_files.iterrows():
        print(f" - {row['filepath']}")
    # Remove missing file paths from the DataFrame
    df = df[df['filepath'].apply(os.path.exists)]
    print(f"After exclusion, {len(df)} labeled images remain.")
else:
    print("All file paths exist.")
 
# Extracting file paths and labels for properly splitting data to appropriate path   
image_paths = df['filepath'].tolist()
labels = df['Class'].tolist()


train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels, test_size=0.3, stratify=labels, random_state=42)

print(f"Training set size: {len(train_paths)} images")
print(f"Temp set size (val + test): {len(temp_paths)} images")

# Second split: Validation and Test (each 15% of total data)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42)

print(f"Validation set size: {len(val_paths)} images")
print(f"Test set size: {len(test_paths)} images")


import shutil

def copy_images(paths, labels, split):
    for path, label in zip(paths, labels):
        filename = os.path.basename(path)
        dest_dir = os.path.join(data_dir, split, label)
        dest_path = os.path.join(dest_dir, filename)
        
        if os.path.exists(dest_path):
            print(f"Skipping copy for '{filename}' as it already exists in '{split}/{label}'.")
            continue  # Skip copying this file
        
        try:
            shutil.copy2(path, dest_dir)  # Using copy2 to preserve metadata
            print(f"Copied '{filename}' to '{split}/{label}'.")
        except shutil.Error as e:
            print(f"Error copying '{path}' to '{dest_dir}': {e}")
        except FileNotFoundError:
            print(f"Destination directory '{dest_dir}' does not exist.")
        except Exception as e:
            print(f"An unexpected error occurred while copying '{filename}': {e}")

# Execute copying for each split
print("\nCopying training images...")
copy_images(train_paths, train_labels, 'train')

print("\nCopying validation images...")
copy_images(val_paths, val_labels, 'val')

print("\nCopying test images...")
copy_images(test_paths, test_labels, 'test')


# Split Summary
print("\nData Splitting Completed:")
print(f" - Training set: {len(train_paths)} images")
print(f" - Validation set: {len(val_paths)} images")
print(f" - Test set: {len(test_paths)} images")

import torchvision.transforms as transforms

# BYOL-specific augmentations
byol_transform = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Classification-specific augmentations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(512, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Validation and Test transforms
val_test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy

class BYOL(nn.Module):
    def __init__(self, base_encoder, feature_dim=2048, projection_dim=256, hidden_dim=4096):
        """
        Initializes the BYOL model.

        Args:
            base_encoder (nn.Module): The base encoder network (e.g., ResNet-50 without the final fc layer).
            feature_dim (int): The dimensionality of the encoder's output features.
            projection_dim (int): The dimensionality of the projection head.
            hidden_dim (int): The dimensionality of the hidden layer in the projector and predictor.
        """
        super(BYOL, self).__init__()
        
        self.feature_dim = feature_dim  # e.g., 2048 for ResNet-50
        
        # Online network
        self.online_encoder = base_encoder
        self.online_projector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Target network: create a deep copy of the online encoder
        self.target_encoder = copy.deepcopy(base_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False  # Freeze target encoder parameters
        
        self.target_projector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )
        for param in self.target_projector.parameters():
            param.requires_grad = False  # Freeze target projector parameters

        # Initialize target network to have the same weights as online network
        self._update_target_network(tau=1.0)

    @torch.no_grad()
    def _update_target_network(self, tau):
        """
        Update target network parameters as an exponential moving average of online network parameters.

        Args:
            tau (float): Momentum parameter for updating the target network.
        """
        for param_o, param_t in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_t.data = tau * param_t.data + (1 - tau) * param_o.data
        for param_o, param_t in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            param_t.data = tau * param_t.data + (1 - tau) * param_o.data

    def forward(self, x1, x2):
        """
        Forward pass for BYOL.

        Args:
            x1 (torch.Tensor): First augmented view of the input batch.
            x2 (torch.Tensor): Second augmented view of the input batch.

        Returns:
            torch.Tensor: Combined BYOL loss for the batch.
        """
        # Online network forward pass for first view
        online_rep1 = self.online_encoder(x1)
        online_proj1 = self.online_projector(online_rep1)
        online_pred1 = self.online_predictor(online_proj1)

        # Online network forward pass for second view
        online_rep2 = self.online_encoder(x2)
        online_proj2 = self.online_projector(online_rep2)
        online_pred2 = self.online_predictor(online_proj2)

        with torch.no_grad():
            # Target network forward pass for first view
            target_rep1 = self.target_encoder(x1)
            target_proj1 = self.target_projector(target_rep1)

            # Target network forward pass for second view
            target_rep2 = self.target_encoder(x2)
            target_proj2 = self.target_projector(target_rep2)

        # Compute BYOL loss
        loss = self.loss_fn(online_pred1, target_proj2) + self.loss_fn(online_pred2, target_proj1)
        
        # Update target network with momentum
        self._update_target_network(tau=0.99)

        return loss

    def loss_fn(self, p, z):
        """
        BYOL loss: 2 - 2 * cosine_similarity

        Args:
            p (torch.Tensor): Predictions from the online predictor.
            z (torch.Tensor): Projections from the target projector.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return 2 - 2 * (p * z).sum(dim=1).mean()

def get_base_encoder():
    """
    Initializes a ResNet-50 encoder without ImageNet pretrained weights.

    Returns:
        nn.Module: ResNet-50 model with the final fc layer replaced by Identity.
    """
    base_encoder = models.resnet50(pretrained=False)  # Do not load ImageNet weights
    feature_dim = base_encoder.fc.in_features  # Typically 2048 for ResNet-50
    base_encoder.fc = nn.Identity()  # Remove the original classification head
    return base_encoder, feature_dim

class BYOLDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        # Convert to RGB if needed
        image = image.convert('RGB')
        
        # Apply the BYOL transform twice to get two different augmented views
        img1 = self.transform(image)
        img2 = self.transform(image)
        
        return img1, img2

# Instantiate the BYOL model
base_encoder, feature_dim = get_base_encoder()
byol_model = BYOL(base_encoder=base_encoder, feature_dim=feature_dim)

# ----------------------------------------------
# Commented Out: Loading Pretrained BYOL Weights
# ----------------------------------------------
# Path to your pretrained BYOL weights
# pretrained_byol_path = 'path_to_pretrained_byol.pth'  # <-- Replace with your actual path

# Load the pretrained BYOL weights
# Note:
# - Ensure that 'pretrained_byol.pth' contains weights matching the BYOL model's architecture.
# - If the pretrained weights include only the online network, set strict=False.
# - Adjust 'map_location' if loading on a different device.
# state_dict = torch.load(pretrained_byol_path, map_location='cpu')
# byol_model.load_state_dict(state_dict, strict=False)  # Set strict=True if all keys match

# Move the model to GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
byol_model = byol_model.to(device)

# Optional: Set the model to training mode
byol_model.train()

# Now, the BYOL model is ready to be trained on your unlabeled data from scratch

# Assuming you have your dataset ready and transformations applied
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Hyperparameters
learning_rate = 1e-3  # You can experiment with this
batch_size = 32  # Adjust based on your dataset size and GPU memory
num_epochs = 100  # Adjust based on your training needs

# Create DataLoader for your dataset
train_dataset = ...  # Your dataset here
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize optimizer
optimizer = optim.Adam(byol_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    byol_model.train()  # Set model to training mode
    total_loss = 0.0
    
    for x1, x2 in train_loader:  # Assuming your DataLoader yields pairs of augmented images
        optimizer.zero_grad()  # Zero the gradients
        
        loss = byol_model(x1, x2)  # Forward pass
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# After training, you can save your model if needed
torch.save(byol_model.state_dict(), 'byol_model.pth')
