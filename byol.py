import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from data_loader import UnlabeledDataset
from byol_model import BYOL  # BYOL implementation

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Load unlabeled data
unlabeled_dataset = UnlabeledDataset('unlabeled_images.xlsx', transform=transform)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)

# Initialize BYOL model and optimizer
byol = BYOL().to(device)
optimizer = torch.optim.Adam(byol.parameters(), lr=3e-4)

# Pretraining loop
for epoch in range(10):
    byol.train()
    total_loss = 0

    for images in unlabeled_loader:
        images = images.to(device)
        loss = byol(images)  # BYOL loss function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        byol.update_target_network()
        total_loss += loss.item()

    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(unlabeled_loader):.4f}')

# Save the pretrained model
torch.save(byol.state_dict(), 'byol_pretrained.pth')