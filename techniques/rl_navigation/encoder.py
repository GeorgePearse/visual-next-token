"""
Semantic Encoder for RL Navigation

Wraps pre-trained vision models (DINOv2, CLIP) to extract semantic features
that are invariant to irrelevant variations (e.g., car color).

Supports two-phase training:
- Phase 1: Frozen encoder provides stable semantic space
- Phase 2: Fine-tune top layers for task-specific features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticEncoder(nn.Module):
    """
    Semantic feature extractor based on pre-trained vision models.

    Handles:
    - Loading pre-trained weights (DINOv2, CLIP)
    - Freezing/unfreezing for two-phase training
    - Extracting patch-level or global features
    - Invariance to low-level variations (color, texture)
    """

    def __init__(
        self,
        model_name="dinov2_vitb14",
        patch_size=14,
        freeze=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model_name: Pre-trained model to use
                - "dinov2_vitb14": DINOv2 ViT-B/14 (recommended)
                - "dinov2_vits14": DINOv2 ViT-S/14 (smaller, faster)
                - "dinov2_vitl14": DINOv2 ViT-L/14 (larger, more capacity)
            patch_size: Patch size of vision transformer
            freeze: Whether to freeze weights initially
            device: Device to load model on
        """
        super().__init__()

        self.model_name = model_name
        self.patch_size = patch_size
        self.device = device

        # Load pre-trained model
        print(f"Loading {model_name}...")
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model = self.model.to(device)
        self.model.eval()  # Always in eval mode for feature extraction

        # Get feature dimension
        self.feature_dim = self.model.embed_dim

        if freeze:
            self.freeze()

        print(f"SemanticEncoder initialized: {self.feature_dim}-dim features")

    def freeze(self):
        """Freeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        print("Encoder frozen")

    def unfreeze_all(self):
        """Unfreeze all model parameters."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("Encoder fully unfrozen")

    def unfreeze_top_layers(self, n_layers=2):
        """
        Unfreeze only the top n transformer blocks.

        Args:
            n_layers: Number of top blocks to unfreeze (default: 2)
        """
        # First ensure everything is frozen
        self.freeze()

        # Unfreeze top n blocks
        for block in self.model.blocks[-n_layers:]:
            for param in block.parameters():
                param.requires_grad = True

        # Also unfreeze final norm layer
        for param in self.model.norm.parameters():
            param.requires_grad = True

        print(f"Unfroze top {n_layers} transformer blocks")

    def forward(self, x):
        """
        Extract semantic features from image patches.

        Args:
            x: Image tensor (B, C, H, W)

        Returns:
            features: (B, num_patches, feature_dim)
        """
        with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.parameters())):
            # Get patch features from DINOv2
            # DINOv2 returns [CLS token, patch tokens]
            features = self.model.forward_features(x)

            # Extract patch tokens (skip CLS token)
            patch_features = features["x_norm_patchtokens"]  # (B, num_patches, feature_dim)

            return patch_features

    def get_patch_features_at_position(self, image, position, patch_radius=1):
        """
        Get semantic features for a local region around a position.

        Args:
            image: Image tensor (C, H, W) or (B, C, H, W)
            position: (row, col) in pixel coordinates
            patch_radius: How many patches around position to extract

        Returns:
            features: Semantic features for the local region
        """
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        # Get all patch features
        with torch.no_grad():
            all_features = self.forward(image)  # (1, num_patches, feature_dim)

        # Convert pixel position to patch position
        row, col = position
        patch_row = row // self.patch_size
        patch_col = col // self.patch_size

        # Calculate patch grid dimensions
        _, _, H, W = image.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size

        # Extract local patch features
        local_features = []
        for dr in range(-patch_radius, patch_radius + 1):
            for dc in range(-patch_radius, patch_radius + 1):
                pr = patch_row + dr
                pc = patch_col + dc

                # Check bounds
                if 0 <= pr < num_patches_h and 0 <= pc < num_patches_w:
                    patch_idx = pr * num_patches_w + pc
                    local_features.append(all_features[0, patch_idx])

        # Average pool local features
        if local_features:
            features = torch.stack(local_features).mean(dim=0)
        else:
            # Fallback to zeros if position is out of bounds
            features = torch.zeros(self.feature_dim, device=self.device)

        return features

    def get_global_features(self, image):
        """
        Get global image features (CLS token).

        Args:
            image: Image tensor (C, H, W) or (B, C, H, W)

        Returns:
            features: Global semantic features
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            features_dict = self.model.forward_features(image)
            cls_token = features_dict["x_norm_clstoken"]  # (B, feature_dim)

        return cls_token.squeeze(0)

    def precompute_patch_features(self, image):
        """
        Precompute all patch features for an image.

        Useful for efficient feature lookup during navigation.

        Args:
            image: Image tensor (C, H, W) or (B, C, H, W)

        Returns:
            patch_features: (num_patches_h, num_patches_w, feature_dim)
            Arranged in spatial grid
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(self.device)

        with torch.no_grad():
            all_features = self.forward(image)  # (1, num_patches, feature_dim)

        # Reshape to spatial grid
        _, _, H, W = image.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size

        patch_features = all_features[0].reshape(num_patches_h, num_patches_w, self.feature_dim)

        return patch_features


def test_encoder():
    """Test the semantic encoder."""
    import numpy as np

    # Create random image
    image = torch.rand(3, 224, 224)

    # Initialize encoder
    encoder = SemanticEncoder(model_name="dinov2_vits14", freeze=True)

    # Test forward pass
    features = encoder.get_patch_features_at_position(image, (112, 112))
    print(f"Patch features shape: {features.shape}")

    # Test precomputation
    all_features = encoder.precompute_patch_features(image)
    print(f"All patch features shape: {all_features.shape}")

    # Test unfreezing
    encoder.unfreeze_top_layers(n_layers=2)

    print("Encoder test passed!")


if __name__ == "__main__":
    test_encoder()
