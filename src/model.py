import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights  # Import ResNet50_Weights
import copy


class SupervisedModel(nn.Module):
    def __init__(self, encoder, feature_dim, num_classes, freeze_layers=None):
        """
        Initializes the supervised model by attaching a classifier head to the encoder.

        Args:
            encoder (nn.Module): The pretrained encoder (from BYOL).
            feature_dim (int): Dimensionality of the encoder's output features.
            num_classes (int): Number of target classes.
            freeze_layers (str, optional): Which layers to freeze:
                - 'all': Freeze entire encoder
                - 'partial': Freeze first few layers
                - None: Don't freeze any layers (default)
        """
        super(SupervisedModel, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Apply freezing strategy
        self.freeze_layers(freeze_layers)

    def freeze_layers(self, freeze_option):
        """
        Freezes specified layers of the encoder.
        
        Args:
            freeze_option (str): Which layers to freeze
        """
        if freeze_option is None:
            return
        
        if freeze_option == 'all':
            # Freeze all encoder layers
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        elif freeze_option == 'partial':
            # Freeze first few layers (typically up to layer3 in ResNet)
            layers_to_freeze = [
                self.encoder.conv1,
                self.encoder.bn1,
                self.encoder.layer1,
                self.encoder.layer2,
                self.encoder.layer3
            ]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x):
        """
        Forward pass for supervised training.

        Args:
            x (Tensor): Input tensor.

        Returns:
            logits (Tensor): Output logits for classification.
        """
        features = self.encoder(x)
        if len(features.shape) > 2:
            features = features.squeeze(-1).squeeze(-1)
        return self.classifier(features)


class BYOL(nn.Module):
    def __init__(
        self, base_encoder, feature_dim=2048, projection_dim=256, hidden_dim=4096
    ):
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
            nn.Linear(hidden_dim, projection_dim),
        )
        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

        # Target network: create a deep copy of the online encoder
        self.target_encoder = copy.deepcopy(base_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False  # Freeze target encoder parameters

        self.target_projector = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        for param in self.target_projector.parameters():
            param.requires_grad = False  # Freeze target projector parameters

        # Initialize target network to have the same weights as online network
        self._update_target_network(tau=1.0)

    @torch.no_grad()
    def _update_target_network(self, tau):
        """
        Update target network parameters deterministically
        Update target network parameters deterministically
        """
        with torch.no_grad():
            for param_o, param_t in zip(
                self.online_encoder.parameters(), self.target_encoder.parameters()
            ):
                param_t.data = tau * param_t.data + (1 - tau) * param_o.data
            for param_o, param_t in zip(
                self.online_projector.parameters(), self.target_projector.parameters()
            ):
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
        loss = self.loss_fn(online_pred1, target_proj2) + self.loss_fn(
            online_pred2, target_proj1
        )

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
        cosine_sim = (p * z).sum(dim=1).mean()
        loss = 2 - 2 * cosine_sim
        return loss


def get_base_encoder(pretrained=True, custom_resnet=False):
    """
    Initializes the ResNet encoder and returns it along with its feature dimension.

    Initializes the ResNet encoder and returns it along with its feature dimension.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet.

        pretrained (bool): If True, returns a model pre-trained on ImageNet.

    Returns:
        encoder (nn.Module): The ResNet encoder without the final fully connected layer.
        feature_dim (int): The dimensionality of the encoder's output features.
        encoder (nn.Module): The ResNet encoder without the final fully connected layer.
        feature_dim (int): The dimensionality of the encoder's output features.
    """

    if custom_resnet:
        # Use the custom ResNet model
        model = models.resnet50(weights = None)
        # Load the state dictionary
        weights_path = 'data/pretrain_res50x1.pth'
        state_dict = torch.load(weights_path, map_location='cpu')

        # Load the weights into the model
        model.load_state_dict(state_dict)
    else:
    # Use the new weights enum for ResNet50
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
        else:
            weights = None

        model = models.resnet50(weights=weights)
    # Get the feature dimension before modifying the model
    feature_dim = model.fc.in_features  # Typically 2048 for ResNet-50

    # Remove the final fully connected layer
    model.fc = nn.Identity()  # Replace with Identity instead of removing

    return model, feature_dim
