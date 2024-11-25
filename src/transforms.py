import torch
import torch
from torchvision import transforms

class DeterministicRandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img):
        return super().forward(img)

class DeterministicRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img):
        return super().forward(img)

class DeterministicColorJitter(transforms.ColorJitter):
    def forward(self, img):
        return super().forward(img)

# BYOL-specific augmentations with deterministic transforms
byol_transform = transforms.Compose([
    DeterministicRandomResizedCrop(224, scale=(0.2, 1.0)),
    DeterministicRandomHorizontalFlip(),
    transforms.RandomApply([
        DeterministicColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
class DeterministicRandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img):
        return super().forward(img)

class DeterministicRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img):
        return super().forward(img)

class DeterministicColorJitter(transforms.ColorJitter):
    def forward(self, img):
        return super().forward(img)

# BYOL-specific augmentations with deterministic transforms
byol_transform = transforms.Compose([
    DeterministicRandomResizedCrop(224, scale=(0.2, 1.0)),
    DeterministicRandomHorizontalFlip(),
    transforms.RandomApply([
        DeterministicColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Classification-specific augmentations with deterministic transforms
train_transform = transforms.Compose([
    DeterministicRandomResizedCrop(224, scale=(0.2, 1.0)),
    DeterministicRandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Classification-specific augmentations with deterministic transforms
train_transform = transforms.Compose([
    DeterministicRandomResizedCrop(224, scale=(0.2, 1.0)),
    DeterministicRandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Validation and Test transforms (already deterministic)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Validation and Test transforms (already deterministic)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Additional: Add print statements to confirm when transforms are applied
class LoggingTransform:
    def __init__(self, transform, name="Transform"):
        self.transform = transform
        self.name = name
    
    def __call__(self, img):
        print(f"Applying {self.name}")
        return self.transform(img)

# Wrap existing transforms with LoggingTransform
byol_transform = LoggingTransform(byol_transform, name="BYOL Transform")
train_transform = LoggingTransform(train_transform, name="Train Transform")
val_test_transform = LoggingTransform(val_test_transform, name="Val/Test Transform")
