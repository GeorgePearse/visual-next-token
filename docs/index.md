# Visual Next Token - RL-Based Image Navigation

Welcome to Visual Next Token! This project implements **curiosity-driven reinforcement learning** for learning semantic paths through images by maximizing prediction error of future visual tokens.

## What is Visual Next Token?

Visual Next Token explores how agents can learn to navigate images by seeking information-dense regions. The core insight: **an agent that maximizes prediction error (not accuracy) will naturally explore semantically meaningful paths** - from cars to roads to sky, following co-occurrence patterns in visual scenes.

## Key Features

### 🧠 RL-Based Image Navigation
- **Curiosity-driven exploration** using prediction error as intrinsic reward
- **Two-phase training**: frozen encoder → fine-tuned encoder
- **Semantic invariance**: DINOv2 features solve the "car color problem"
- **Exponential distance weighting** for multi-step lookahead planning

### 🔬 Multiple Intrinsic Motivation Methods
- **ICM (Intrinsic Curiosity Module)**: Forward dynamics prediction
- **RND (Random Network Distillation)**: Fixed target network prediction

### 🎯 Production-Ready Components
- PPO policy optimization with GAE
- Hierarchical action spaces (base + jump/scout actions)
- Comprehensive training and visualization tools

## Quick Example

```python
from techniques.rl_navigation import RLTrainer

# Train RL navigator on your image
trainer = RLTrainer(
    image=my_image,
    encoder_name="dinov2_vits14",
    phase1_episodes=10000,  # Frozen encoder
    phase2_episodes=5000,   # Fine-tuned encoder
)

trainer.train()
```

```bash
# Visualize learned paths
python experiments/visualize_rl_paths.py \
    --checkpoint checkpoints/rl_navigator/final.pt \
    --image my_image.jpg \
    --n_episodes 5
```

## The Core Insight

Traditional approaches predict **what** pixels come next. We flip this:

!!! success "Our Approach"
    **Maximize prediction ERROR** → Agent seeks surprising, information-dense regions

    - High prediction error = novel/informative region
    - Agent learns paths like: car → road + sky → buildings
    - Semantic features (DINOv2) ensure car color doesn't matter

!!! failure "Traditional Approach"
    **Maximize prediction ACCURACY** → Agent seeks boring, predictable regions

    - High accuracy = uniform sky, blank walls
    - No incentive to explore semantically rich areas

## Project Structure

```
image-ssl/
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
└── references/
    └── rl_navigation/          # Key papers with summaries
```

## Research Foundation

Our implementation builds on five key papers:

1. **[ICM](references/rl-navigation/01_curiosity_driven_exploration_ICM.md)** - Curiosity-driven exploration (Pathak et al., 2017)
2. **[RND](references/rl-navigation/02_random_network_distillation_RND.md)** - Random network distillation (Burda et al., 2018)
3. **[PPO](references/rl-navigation/03_proximal_policy_optimization_PPO.md)** - Policy optimization (Schulman et al., 2017)
4. **[DINOv2](references/rl-navigation/04_dinov2_visual_features.md)** - Semantic features (Oquab et al., 2023)
5. **[GAE](references/rl-navigation/05_generalized_advantage_estimation_GAE.md)** - Advantage estimation (Schulman et al., 2015)

## Next Steps

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Quick Start__

    ---

    Get up and running in minutes

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :fontawesome-solid-brain:{ .lg .middle } __RL Navigation__

    ---

    Deep dive into curiosity-driven navigation

    [:octicons-arrow-right-24: Architecture](techniques/rl-navigation/architecture.md)

-   :material-file-document:{ .lg .middle } __Research Papers__

    ---

    Understand the theoretical foundations

    [:octicons-arrow-right-24: References](references/index.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    Detailed API documentation

    [:octicons-arrow-right-24: API](api/rl-navigation.md)

</div>
