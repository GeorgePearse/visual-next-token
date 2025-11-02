# Research References

This section contains detailed summaries of the key papers that form the theoretical foundation of our RL-based image navigation implementation.

## Overview

Our approach synthesizes ideas from five major research areas:

1. **Curiosity-Driven RL** - Using prediction error as intrinsic motivation
2. **Semantic Visual Features** - Self-supervised learning for robust representations
3. **Policy Optimization** - Stable, sample-efficient RL algorithms
4. **Advantage Estimation** - Variance reduction in policy gradients
5. **Hierarchical RL** - Extended action spaces for long-range planning

## Key Papers

### RL Navigation Foundation

These five papers directly inform our implementation:

| Paper | Year | Contribution | Our Implementation |
|-------|------|--------------|-------------------|
| [ICM](rl_navigation/01_curiosity_driven_exploration_ICM.md) | 2017 | Curiosity via prediction error | `ForwardDynamicsModel` |
| [RND](rl_navigation/02_random_network_distillation_RND.md) | 2018 | Random network distillation | `RNDIntrinsicMotivation` |
| [PPO](rl_navigation/03_proximal_policy_optimization_PPO.md) | 2017 | Clipped policy optimization | `PPOTrainer` |
| [DINOv2](rl_navigation/04_dinov2_visual_features.md) | 2023 | Semantic visual features | `SemanticEncoder` |
| [GAE](rl_navigation/05_generalized_advantage_estimation_GAE.md) | 2015 | Advantage estimation | `compute_gae()` |

### Quick Reference

#### Intrinsic Motivation

**When to use ICM vs RND?**

- **ICM (Forward Dynamics):**
  - Action-relevant exploration
  - Learn what actions lead where
  - More sample efficient
  - Can be less stable

- **RND (Random Network Distillation):**
  - State-space novelty
  - More stable training
  - Simpler implementation
  - Less action-focused

#### Feature Learning

**Why DINOv2 over other encoders?**

DINOv2 provides:
- ✅ Pre-trained on diverse data (142M images)
- ✅ Semantic invariance (car color problem)
- ✅ Patch-level features (natural for navigation)
- ✅ Multiple model sizes (21M - 1.1B params)
- ✅ Strong zero-shot transfer

Alternatives considered:
- CLIP: Language-biased, less dense features
- ResNet: Lower-level features, less semantic
- MAE: Reconstruction-focused, not semantic

#### Policy Optimization

**Why PPO over other methods?**

PPO balances:
- ✅ Sample efficiency (vs A3C, REINFORCE)
- ✅ Stability (vs TRPO complexity)
- ✅ Simplicity (vs SAC, TD3)
- ✅ Works with discrete actions

For image navigation:
- Episodes can be 500+ steps
- Need stable learning (don't want collapse)
- Discrete action space (8 directions)
- PPO is battle-tested choice

## How Papers Relate

```
Image → DINOv2 → Features
         [#4]        │
                     │
         ┌───────────┴──────────┐
         │                      │
         ▼                      ▼
    Policy (π)           Forward Model (P)
    PPO [#3]             ICM/RND [#1,#2]
         │                      │
         │                      ▼
         │              Prediction Error
         │              (Intrinsic Reward)
         │                      │
         └──────────┬───────────┘
                    ▼
              PPO Update
              with GAE [#5]
```

### The Critical Insights

**1. Flip the Reward ([ICM](rl_navigation/01_curiosity_driven_exploration_ICM.md))**

Don't predict accurately → predict **poorly**!

High prediction error = novel/surprising = informative

**2. Semantic Space ([DINOv2](rl_navigation/04_dinov2_visual_features.md))**

Predict in feature space, not pixel space

Red car ≈ Blue car in embeddings, ≠ in pixels

**3. Stable Updates ([PPO](rl_navigation/03_proximal_policy_optimization_PPO.md))**

Clip policy updates to prevent catastrophic collapse

```python
ratio = π_new / π_old
clipped_ratio = clip(ratio, 1-ε, 1+ε)
loss = -min(ratio * A, clipped_ratio * A)
```

**4. Two-Phase Training (Novel)**

Phase 1: Frozen encoder → stable learning
Phase 2: Fine-tune encoder → task adaptation

**5. Gradient Decoupling (Novel)**

Policy uses `features.detach()` → no policy gradients into encoder

Encoder only updated via predictor loss

## Additional Reading

### Hierarchical RL

For jump/scout actions extension:

- Sutton et al., "Between MDPs and semi-MDPs" (1999)
- Kulkarni et al., "Hierarchical Deep RL" (2016)
- Bacon et al., "The Option-Critic Architecture" (2017)

### Exploration in RL

Related intrinsic motivation methods:

- Schmidhuber, "Formal Theory of Creativity" (2010)
- Stadie et al., "Incentivizing Exploration" (2015)
- Badia et al., "Never Give Up" (2020)

### Self-Supervised Vision

Other feature learning approaches:

- Chen et al., "SimCLR" (2020)
- He et al., "Masked Autoencoders" (2021)
- Caron et al., "DINOv1" (2021)

## Citation Guide

When citing our work or building upon it, please cite the relevant foundation papers.

**Minimal citation** (just ICM + DINOv2):
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

**Complete citation** (all five papers):

See individual paper pages for full BibTeX.

## Python RL Frameworks

When implementing RL systems, you can choose from several mature frameworks. Here are the most popular:

### Production-Ready Frameworks

| Framework | Strengths | Use Case | Algorithms |
|-----------|-----------|----------|------------|
| **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** | Easy to use, well-documented, PyTorch-based | Best for getting started, research | PPO, A2C, SAC, TD3, DQN |
| **[Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)** | Distributed training, production-ready, scalable | Large-scale deployments | PPO, IMPALA, APPO, DQN, SAC |
| **[CleanRL](https://github.com/vwxyzjn/cleanrl)** | Single-file implementations, highly readable | Learning RL, research, modifications | PPO, DQN, SAC, TD3 |
| **[Tianshou](https://github.com/thu-ml/tianshou)** | Fast, modular, elegant API | Research, custom algorithms | PPO, SAC, DQN, DDPG, A2C |

### Specialized Frameworks

| Framework | Focus | Best For |
|-----------|-------|----------|
| **[Gymnasium](https://gymnasium.farama.org/)** | Environment interface (successor to OpenAI Gym) | Creating custom environments |
| **[PettingZoo](https://pettingzoo.farama.org/)** | Multi-agent RL | Competitive/cooperative agents |
| **[Sample Factory](https://github.com/alex-petrenko/sample-factory)** | High-throughput training | Very fast environments (games) |
| **[RLlib Trainer](https://docs.ray.io/en/latest/rllib/index.html)** | Hyperparameter tuning at scale | Large experiments, tuning |

### Our Implementation

**Visual Next Token** uses a **custom PPO implementation** because:

- ✅ Full control over gradient flow (detached features for policy)
- ✅ Tight integration with DINOv2 encoder
- ✅ Two-phase training (freeze → fine-tune encoder)
- ✅ Custom reward computation (rolling-window accuracy)
- ✅ Educational value (understand PPO internals)

**If you want to use an existing framework**, here's how you'd adapt:

#### Stable-Baselines3 Example

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

# 1. Wrap environment in Gymnasium interface
class GymImageNavEnv(gym.Env):
    def __init__(self, image, encoder, predictor):
        self.env = ImageNavigationEnv(image, encoder, predictor)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(encoder.feature_dim,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(8)

    def reset(self, seed=None):
        state = self.env.reset()
        return state["features"].cpu().numpy(), {}

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        return next_state["features"].cpu().numpy(), reward, done, False, info

# 2. Train with SB3
env = GymImageNavEnv(image, encoder, predictor)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4)
model.learn(total_timesteps=100000)
```

#### CleanRL Example

CleanRL's single-file implementations are excellent for understanding and modifying algorithms. You could adapt [cleanrl/ppo.py](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py) by:

1. Replacing environment with `ImageNavigationEnv`
2. Modifying reward computation for rolling-window accuracy
3. Adding two-phase training logic

### Choosing a Framework

**Use Stable-Baselines3 if**:
- You want quick experiments with standard algorithms
- You need reliable, battle-tested implementations
- Documentation and community support are important

**Use CleanRL if**:
- You want to understand algorithm internals
- You plan to modify or extend algorithms
- You prefer minimal dependencies

**Use Ray RLlib if**:
- You need distributed training across many machines
- You're running hyperparameter sweeps
- You're deploying to production

**Use Custom Implementation (like ours) if**:
- You need tight integration with custom components
- You want full control over training dynamics
- You're doing research on novel RL methods

## Next Steps

<div class="grid cards" markdown>

-   :material-file-document:{ .lg .middle } __Read the Papers__

    ---

    Detailed summaries with code connections

    [:octicons-arrow-right-24: RL Navigation Papers](rl_navigation/01_curiosity_driven_exploration_ICM.md)

-   :fontawesome-solid-brain:{ .lg .middle } __Understand the Architecture__

    ---

    How papers combine into working system

    [:octicons-arrow-right-24: Architecture](../techniques/rl_navigation/architecture.md)

-   :material-code-braces:{ .lg .middle } __See the Code__

    ---

    Implementation details and API

    [:octicons-arrow-right-24: API Reference](../api/rl_navigation.md)

</div>
