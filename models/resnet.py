"""
ResNet backbone for SSL experiments.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


class ResNetBackbone(nn.Module):
    """
    ResNet backbone with projection head for SSL.
    """

    def __init__(self, arch="resnet50", proj_dim=128, pretrained=False):
        """
        Args:
            arch: Architecture name ('resnet18', 'resnet50')
            proj_dim: Dimension of projection head output
            pretrained: Whether to use ImageNet pretrained weights
        """
        super().__init__()

        # Load base model
        if arch == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = resnet18(weights=weights)
            feat_dim = 512
        elif arch == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.encoder = resnet50(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Remove the final FC layer
        self.encoder.fc = nn.Identity()

        # Projection head (typically 2-3 layer MLP)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True), nn.Linear(feat_dim, proj_dim)
        )

    def forward(self, x, return_features=False):
        """
        Args:
            x: Input images
            return_features: If True, return both features and projections

        Returns:
            Projection head output, optionally with backbone features
        """
        features = self.encoder(x)
        projections = self.projection(features)

        if return_features:
            return features, projections
        return projections


def build_resnet(arch="resnet50", proj_dim=128, pretrained=False):
    """
    Factory function to build ResNet backbone.

    Args:
        arch: Architecture name
        proj_dim: Projection dimension
        pretrained: Use ImageNet pretrained weights

    Returns:
        ResNetBackbone model
    """
    return ResNetBackbone(arch=arch, proj_dim=proj_dim, pretrained=pretrained)
