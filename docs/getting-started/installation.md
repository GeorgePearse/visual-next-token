# Installation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended, but CPU works)

## Install from Source

```bash
# Clone the repository
git clone https://github.com/georgepearse/image-ssl.git
cd image-ssl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

The project requires:

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.7.0
matplotlib>=3.7.0
```

For RL navigation specifically:

- **DINOv2**: Loaded automatically via `torch.hub`
- **Gym**: For RL environment interface (optional)

## Verify Installation

```python
# Test basic imports
from techniques.rl_navigation import (
    RLTrainer,
    SemanticEncoder,
    NavigationPolicy,
)

print("✓ Installation successful!")
```

## GPU Support

Check if CUDA is available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
```

The code automatically selects GPU if available, otherwise falls back to CPU.

## Download Pre-trained Models

DINOv2 models are downloaded automatically on first use via `torch.hub`. Available models:

| Model | Parameters | Feature Dim | Download Size |
|-------|-----------|-------------|---------------|
| `dinov2_vits14` | 21M | 384 | ~84 MB |
| `dinov2_vitb14` | 86M | 768 | ~330 MB |
| `dinov2_vitl14` | 300M | 1024 | ~1.1 GB |
| `dinov2_vitg14` | 1.1B | 1536 | ~4.2 GB |

First run will download the selected model to `~/.cache/torch/hub/`.

## Next Steps

- [Quick Start Guide](quickstart.md) - Run your first RL navigation experiment
- [RL Navigation Architecture](../techniques/rl-navigation/architecture.md) - Understand the system design
