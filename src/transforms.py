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

class BYOLTransform:
    """
    BYOL-specific transformation class that generates two differently augmented views of an image.
    """
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),  # Broader scale range
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1
                )
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=23,
                    sigma=(0.1, 2.0)
                )
            ], p=1.0),  # Always apply blur to one branch
            transforms.RandomSolarize(threshold=128, p=0.1),  # Additional augmentation
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def __call__(self, x):
        """
        Args:
            x (PIL Image): Input image

        Returns:
            tuple: Two differently augmented versions of the input image
        """
        return self.transform(x), self.transform(x)

# Classification-specific augmentations with deterministic transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2)
])

# Validation and Test transforms (already deterministic)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
