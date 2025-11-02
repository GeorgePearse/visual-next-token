# RL Navigation: Key Papers and References

This directory contains detailed summaries of the foundational papers used in our RL-based image navigation implementation.

## Overview

Our approach combines several key techniques from recent RL and vision research:

1. **Curiosity-Driven Exploration** - Intrinsic motivation via prediction error
2. **Semantic Feature Learning** - DINOv2 for appearance-invariant representations
3. **Policy Optimization** - PPO for stable, sample-efficient learning
4. **Advantage Estimation** - GAE for variance reduction

## Papers

### 1. [Curiosity-driven Exploration by Self-supervised Prediction (ICM)](01_curiosity_driven_exploration_ICM.md)
**Pathak et al., ICML 2017**

- **Core Idea:** Use prediction error as intrinsic reward signal
- **Key Innovation:** Learn features via inverse dynamics to focus on agent-relevant aspects
- **In Our Code:** `ForwardDynamicsModel` class (forward_dynamics.py)

**Why it matters:** Solves the exploration problem - agent seeks information-dense regions by maximizing prediction error in feature space.

### 2. [Exploration by Random Network Distillation (RND)](02_random_network_distillation_RND.md)
**Burda et al., 2018**

- **Core Idea:** Predict fixed random network outputs for intrinsic motivation
- **Key Innovation:** Simpler than ICM, no action conditioning needed
- **In Our Code:** `RNDIntrinsicMotivation` class (forward_dynamics.py)

**Why it matters:** Alternative to ICM with more stable training. Useful when action-conditioned prediction is difficult.

### 3. [Proximal Policy Optimization (PPO)](03_proximal_policy_optimization_PPO.md)
**Schulman et al., 2017**

- **Core Idea:** Clip policy updates to prevent catastrophic collapse
- **Key Innovation:** Trust region benefits with first-order optimization
- **In Our Code:** `PPOTrainer` class (policy.py)

**Why it matters:** The workhorse policy learning algorithm. Enables multiple epochs of minibatch updates on collected experience.

### 4. [DINOv2: Learning Robust Visual Features without Supervision](04_dinov2_visual_features.md)
**Oquab et al., 2023**

- **Core Idea:** Self-supervised vision transformer trained on 142M curated images
- **Key Innovation:** Features work across domains without fine-tuning
- **In Our Code:** `SemanticEncoder` class (encoder.py)

**Why it matters:** Solves the "car color problem" - provides semantic features invariant to low-level appearance variations. Red car and blue car → same embedding.

### 5. [Generalized Advantage Estimation (GAE)](05_generalized_advantage_estimation_GAE.md)
**Schulman et al., 2015**

- **Core Idea:** Balance bias-variance tradeoff in advantage estimation
- **Key Innovation:** Exponentially-weighted n-step advantages (analogous to TD(λ))
- **In Our Code:** `PPOTrainer.compute_gae()` (policy.py:190-223)

**Why it matters:** Reduces variance in policy gradients while controlling bias. Critical for stable learning in our long-horizon image navigation task.

## How They Fit Together

```
┌─────────────────────────────────────────────────────┐
│                    Image (RGB)                       │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  DINOv2 Encoder (E)  │ ◄── Paper #4
          │  Semantic Features   │
          └──────────┬───────────┘
                     │
           ┌─────────┴─────────┐
           │                   │
           ▼                   ▼
    ┌─────────────┐   ┌──────────────────┐
    │ Policy (π)  │   │ Forward Dynamics │ ◄── Paper #1 or #2
    │   PPO       │   │  (P) or RND      │     (ICM or RND)
    └──────┬──────┘   └────────┬─────────┘
           │                   │
           │                   ▼
           │          ┌─────────────────┐
           │          │ Prediction Error│
           │          │ (Intrinsic      │
           │          │  Reward)        │
           │          └────────┬────────┘
           │                   │
           ▼                   ▼
    ┌────────────────────────────────┐
    │   PPO Update with GAE          │ ◄── Papers #3 & #5
    │   - Clipped objective          │
    │   - Advantage estimation       │
    └────────────────────────────────┘
```

## Implementation Architecture

### Phase 1: Frozen Encoder (10k episodes)
- DINOv2 features frozen (pre-trained)
- Policy learns navigation with stable semantic space
- Forward model/RND learns to predict transitions

### Phase 2: Fine-Tuned Encoder (5k episodes)
- Unfreeze top 2 transformer layers
- Very small learning rate (1e-5)
- Encoder adapts to task-specific features
- Critical: Policy uses `z.detach()` to prevent gradient corruption

### Key Design Decisions

| Decision | Papers | Rationale |
|----------|--------|-----------|
| Prediction error as reward | #1, #2 | Encourages seeking information-dense regions |
| Semantic features (not pixels) | #4 | Invariance to irrelevant variations (car color) |
| Two-phase training | #4 | Stable features first, then task adaptation |
| PPO policy learning | #3 | Sample efficiency + stability |
| GAE advantages | #5 | Variance reduction for long episodes |
| Exponential distance weighting | Novel | Multi-step lookahead planning |

## Additional Reading

### Hierarchical RL (for jump actions extension)
- **Options Framework:** Sutton et al., "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning" (1999)
- **Hierarchical DQN:** Kulkarni et al., "Hierarchical Deep Reinforcement Learning" (2016)

### Intrinsic Motivation
- **Empowerment:** Mohamed & Rezende, "Variational Information Maximisation for Intrinsically Motivated Reinforcement Learning" (2015)
- **NGU:** Badia et al., "Never Give Up: Learning Directed Exploration Strategies" (2020)

### Vision Transformers
- **ViT:** Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (2020)
- **DINO (v1):** Caron et al., "Emerging Properties in Self-Supervised Vision Transformers" (2021)

## Citation

If you use this implementation or build upon these ideas, please cite the relevant papers:

```bibtex
@inproceedings{pathak2017curiosity,
  title={Curiosity-driven exploration by self-supervised prediction},
  author={Pathak, Deepak and Agrawal, Pulkit and Efros, Alexei A and Darrell, Trevor},
  booktitle={ICML},
  year={2017}
}

@article{burda2018exploration,
  title={Exploration by random network distillation},
  author={Burda, Yuri and Edwards, Harrison and Storkey, Amos and Klimov, Oleg},
  journal={arXiv preprint arXiv:1810.12894},
  year={2018}
}

@article{schulman2017proximal,
  title={Proximal policy optimization algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017}
}

@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}

@article{schulman2015high,
  title={High-dimensional continuous control using generalized advantage estimation},
  author={Schulman, John and Moritz, Philipp and Levine, Sergey and Jordan, Michael and Abbeel, Pieter},
  journal={arXiv preprint arXiv:1506.02438},
  year={2015}
}
```
