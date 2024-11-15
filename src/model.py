import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights  # Import ResNet50_Weights
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

def get_base_encoder(pretrained=True):
    """
    Initializes a ResNet-50 encoder.

    Args:
        pretrained (bool): If True, loads ImageNet pretrained weights.

    Returns:
        nn.Module: ResNet-50 model with the final fc layer replaced by Identity.
        int: Feature dimension of the encoder.
    """
    if pretrained:
        base_encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)  # Updated parameter
    else:
        base_encoder = models.resnet50(weights=None)  # No pretrained weights
    feature_dim = base_encoder.fc.in_features  # Typically 2048 for ResNet-50
    base_encoder.fc = nn.Identity()  # Remove the original classification head
    return base_encoder, feature_dim
