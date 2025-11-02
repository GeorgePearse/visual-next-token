# Visual Next Token - RL-Based Image Navigation

Curiosity-driven reinforcement learning for learning semantic paths through images by maximizing prediction error of future visual tokens.

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://georgepearse.github.io/visual-next-token/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Visual Next Token implements a novel approach to image navigation where an agent learns to explore images by **maximizing prediction error** (not accuracy) in semantic feature space. This curiosity-driven approach forces the agent to seek information-dense regions, naturally following semantic co-occurrence patterns like car → road → sky.

### Key Insight

Traditional approaches predict **what** comes next. We flip this:

- ❌ **Traditional**: Maximize prediction accuracy → agent seeks boring, predictable regions
- ✅ **Our approach**: Maximize prediction error → agent seeks surprising, information-dense regions

### The "Car Color Problem"

**Challenge**: Cars can be any color - pixel-level prediction penalizes irrelevant variations.

**Solution**: Use DINOv2 semantic features where red car ≈ blue car in embedding space, but car ≠ road.

## Quick Start

```bash
# Clone repository
git clone https://github.com/georgepearse/visual-next-token.git
cd visual-next-token

# Install dependencies
pip install torch torchvision numpy opencv-python matplotlib

# Train RL navigator (quick test)
python experiments/train_rl_navigator.py --config quick_test

# Visualize learned paths
python experiments/visualize_rl_paths.py \
    --checkpoint checkpoints/rl_navigator/final.pt \
    --n_episodes 5
```

## Features

### 🧠 RL-Based Image Navigation
- **Curiosity-driven exploration** using prediction error as intrinsic reward
- **Two-phase training**: frozen encoder → fine-tuned encoder
- **Semantic invariance**: DINOv2 features solve appearance variation issues
- **Exponential distance weighting** for multi-step lookahead planning

### 🔬 Multiple Intrinsic Motivation Methods
- **ICM (Intrinsic Curiosity Module)**: Forward dynamics prediction
- **RND (Random Network Distillation)**: Fixed target network prediction

### 🎯 Production-Ready Components
- PPO policy optimization with GAE
- Hierarchical action spaces (base + jump/scout actions)
- Comprehensive training and visualization tools
- Complete documentation with paper summaries

## Architecture

```
Image (RGB) → DINOv2 Encoder → Semantic Features (z)
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
          Navigation Policy (π)            Forward Dynamics (P)
              PPO + GAE                    ICM or RND
                    │                                   │
                    │                                   ▼
                    │                          Prediction Error
                    │                         (Intrinsic Reward)
                    └──────────┬────────────────────────┘
                               ▼
                         PPO Update
```

## Project Structure

```
visual-next-token/
├── techniques/
│   └── rl_navigation/          # RL navigation implementation
│       ├── encoder.py          # DINOv2 semantic encoder
│       ├── environment.py      # MDP for image navigation
│       ├── policy.py           # PPO actor-critic
│       ├── forward_dynamics.py # ICM / RND
│       ├── trainer.py          # Two-phase training
│       ├── extensions.py       # Jump/scout actions
│       └── config.py           # Hyperparameters
├── experiments/
│   ├── train_rl_navigator.py  # Training script
│   └── visualize_rl_paths.py  # Visualization
├── references/
│   └── rl_navigation/          # Key papers with summaries
├── docs/                       # MkDocs documentation
└── README.md
```

## Documentation

Comprehensive documentation available at: **https://georgepearse.github.io/visual-next-token/**

Includes:
- 📚 Detailed architecture explanations
- 🚀 Quick start guides
- 📄 Research paper summaries with code connections
- 🔧 API reference

## Configuration Presets

| Config | Phase 1 | Phase 2 | Use Case |
|--------|---------|---------|----------|
| `quick_test` | 100 | 50 | Testing/debugging |
| `default` | 10,000 | 5,000 | Standard training |
| `rnd` | 10,000 | 5,000 | Use RND instead of ICM |
| `long` | 20,000 | 10,000 | Extended training with larger model |

## Research Foundation

Our implementation builds on five key papers:

1. **[ICM](references/rl_navigation/01_curiosity_driven_exploration_ICM.md)** - Curiosity-driven exploration (Pathak et al., 2017)
2. **[RND](references/rl_navigation/02_random_network_distillation_RND.md)** - Random network distillation (Burda et al., 2018)
3. **[PPO](references/rl_navigation/03_proximal_policy_optimization_PPO.md)** - Policy optimization (Schulman et al., 2017)
4. **[DINOv2](references/rl_navigation/04_dinov2_visual_features.md)** - Semantic features (Oquab et al., 2023)
5. **[GAE](references/rl_navigation/05_generalized_advantage_estimation_GAE.md)** - Advantage estimation (Schulman et al., 2015)

## Example Usage

```python
from techniques.rl_navigation import RLTrainer
import cv2

# Load image
image = cv2.imread("my_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create trainer
trainer = RLTrainer(
    image=image,
    encoder_name="dinov2_vits14",
    phase1_episodes=10000,  # Frozen encoder
    phase2_episodes=5000,   # Fine-tuned encoder
    use_rnd=False,          # Use ICM (or True for RND)
)

# Train
trainer.train()
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- OpenCV
- Matplotlib

DINOv2 models are downloaded automatically via `torch.hub` on first use.

## Citation

If you use this work, please cite the foundational papers:

```bibtex
@inproceedings{pathak2017curiosity,
  title={Curiosity-driven exploration by self-supervised prediction},
  author={Pathak, Deepak and Agrawal, Pulkit and Efros, Alexei A and Darrell, Trevor},
  booktitle={ICML},
  year={2017}
}

@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

See [References](https://georgepearse.github.io/visual-next-token/references/) for complete citation information.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please see our [documentation](https://georgepearse.github.io/visual-next-token/) for more information.
