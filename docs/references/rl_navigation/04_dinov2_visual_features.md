# DINOv2: Learning Robust Visual Features without Supervision

## Paper Information

**arXiv ID:** 2304.07193
**Field:** Computer Science > Computer Vision and Pattern Recognition
**Submitted:** April 14, 2023 (v1)
**Last Revised:** February 2, 2024 (v2)
**DOI:** https://doi.org/10.48550/arXiv.2304.07193

## Authors

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski

## Abstract

The paper demonstrates that "existing pretraining methods, especially self-supervised methods, can produce features that work across image distributions and tasks without finetuning if trained on enough curated data." The researchers combine established techniques to scale pretraining efforts and introduce methods for accelerating and stabilizing large-scale training.

## Key Contributions

1. **Dataset Pipeline:** Created an automatic pipeline for building a diverse, curated image dataset rather than relying on unstructured data typically used in self-supervised learning

2. **Model Scale:** Trained a Vision Transformer with one billion parameters and distilled it into smaller, more practical models

3. **Performance:** Achieved results surpassing OpenCLIP across most benchmarks at both image and pixel levels

4. **Technical Optimizations:** Implemented multiple techniques focused on improving training efficiency and stability at scale

## Resources

- **arXiv:** https://arxiv.org/abs/2304.07193
- **Project Page:** https://dinov2.metademolab.com/
- **GitHub:** https://github.com/facebookresearch/dinov2
- **Models:** Available via torch.hub

## Relevance to Our Implementation

DINOv2 is the **semantic encoder backbone** that solves the critical "car color problem" - the need for features invariant to low-level variations. Implementation in techniques/rl_navigation/encoder.py.

### The Car Color Problem

**Problem:** Pixel-level prediction penalizes irrelevant variations
- Red car vs Blue car → huge pixel difference
- Model shouldn't be penalized for unpredictable color

**Solution:** Semantic feature space
- Red car → `embedding_car`
- Blue car → `embedding_car`
- Prediction in semantic space, not pixel space

### DINOv2 in Our Code

```python
class SemanticEncoder(nn.Module):
    def __init__(self, model_name="dinov2_vits14", freeze=True):
        # Load pre-trained DINOv2
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name
        )
        self.feature_dim = self.model.embed_dim  # 384/768/1024

        if freeze:
            self.freeze()  # Phase 1: frozen features
```

### Available Models

| Model | Parameters | Feature Dim | Patch Size |
|-------|-----------|-------------|------------|
| `dinov2_vits14` | 21M | 384 | 14×14 |
| `dinov2_vitb14` | 86M | 768 | 14×14 |
| `dinov2_vitl14` | 300M | 1024 | 14×14 |
| `dinov2_vitg14` | 1.1B | 1536 | 14×14 |

### Two-Phase Training Strategy

Our implementation uses a critical two-phase approach:

**Phase 1: Frozen Encoder (10k episodes)**
```python
self.encoder.freeze()  # All parameters frozen
# Stable semantic space
# Policy and predictor learn with fixed features
```

**Phase 2: Fine-tuned Encoder (5k episodes)**
```python
self.encoder.unfreeze_top_layers(n_layers=2)  # Last 2 transformer blocks
# encoder_lr = 1e-5  # Very small!
# Task-specific feature adaptation
```

### Why This Matters

1. **Semantic Invariance:** DINOv2 features capture object identity, not appearance
   - Different colored cars → similar features
   - Different car poses → similar features
   - Car vs road vs sky → different features

2. **Zero-Shot Transfer:** Pre-trained on 142M images
   - Works on any image domain
   - No task-specific training needed for Phase 1

3. **Patch-Level Features:** Natural grid structure for image navigation
   ```python
   # Image: 224×224 pixels
   # DINOv2 patches: 16×16 grid (patch_size=14)
   # Features: (16, 16, 384) for vits14
   ```

### Feature Extraction (encoder.py:189-226)

```python
def get_patch_features_at_position(self, image, position, patch_radius=1):
    """Get local patch features around a position."""
    # Convert pixel position → patch coordinates
    patch_row = position[0] // self.patch_size
    patch_col = position[1] // self.patch_size

    # Extract local neighborhood
    features = self.patch_features[
        max(0, patch_row - patch_radius):patch_row + patch_radius + 1,
        max(0, patch_col - patch_radius):patch_col + patch_radius + 1,
    ]

    # Average pool → single feature vector
    return features.mean(dim=(0, 1))
```

### Co-Training Architecture

```
Image → DINOv2 → Features (z) → Forward Model (P) → Predicted Features (ẑ)
                     ↓                                         ↓
                 Policy (π)                            Prediction Error
                     ↓                                         ↓
                  Action                              Intrinsic Reward
```

Critical: Policy uses `z.detach()` to prevent policy gradients from corrupting encoder features!

```python
# trainer.py:203
with torch.no_grad():
    action, log_prob, value = self.policy.act(features.detach())
```

Encoder only updated via predictor gradients in Phase 2.
