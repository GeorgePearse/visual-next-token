# RL-Based Image Navigation

## Overview

RL-based image navigation casts the problem of learning semantic paths through images as a **curiosity-driven reinforcement learning** task. An agent learns to navigate by maximizing prediction error in a learned semantic feature space.

## The Core Idea

### Traditional Approach ❌

Most approaches try to **predict the next pixel/patch accurately**:

```
Reward = -||predicted_pixels - actual_pixels||²
```

**Problem:** Agent seeks predictable, boring regions (uniform sky, blank walls)

### Our Approach ✅

We flip the reward: **maximize prediction ERROR**:

```
Reward = ||predicted_features - actual_features||²
```

**Result:** Agent seeks surprising, information-dense regions!

## Why This Works

The key insight: **prediction error correlates with semantic information content**.

Consider navigating from a car:

| Next Region | Prediction Error | Information Content |
|-------------|-----------------|---------------------|
| More of same car | Low | Low (redundant) |
| Road beneath | High | High (new object type) |
| Sky above | High | High (new context) |
| Another car | High | High (co-occurrence pattern) |

The agent learns that **high prediction error = interesting semantic transitions**.

## The "Car Color Problem"

A critical challenge: **cars can be any color**.

### Problem

If we predict in pixel space:
- Red car → Blue car: HUGE prediction error
- Model is penalized for unpredictable but irrelevant variation

### Solution: Semantic Features

Use DINOv2 to map pixels → semantic space:

```python
# Pixel space (❌)
red_car = [255, 0, 0, ...]  # 3×224×224 = 150,528 dims
blue_car = [0, 0, 255, ...] # Very different!

# Semantic space (✅)
encoder(red_car) = [0.12, -0.34, ...]  # 384 dims
encoder(blue_car) = [0.11, -0.33, ...] # Nearly identical!
```

DINOv2 features are **invariant to appearance**, capturing object identity:
- Different colored cars → similar embeddings
- Car vs road → different embeddings

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                    RGB Image                         │
│                  (H × W × 3)                         │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Semantic Encoder    │
          │     (DINOv2)         │
          │  Frozen → Fine-tuned │
          └──────────┬───────────┘
                     │
           ┌─────────┴─────────┐
           │                   │
           │    Features (z)   │ ◄── Detach for policy!
           │                   │
           └─────────┬─────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
  ┌─────────────┐     ┌──────────────────┐
  │   Policy    │     │ Forward Dynamics │
  │    (π)      │     │      (P)         │
  │  Actor-     │     │  Predict next    │
  │  Critic     │     │  features from   │
  │             │     │  current + action│
  └──────┬──────┘     └────────┬─────────┘
         │                     │
         │ Action              │ Predicted z'
         ▼                     ▼
  ┌────────────────────────────────────┐
  │   Environment Step                 │
  │   - Execute action (move)          │
  │   - Get actual next features (z')  │
  │   - Compute prediction error:      │
  │     reward = ||P(z,a) - z'||²     │
  └────────┬───────────────────────────┘
           │
           ▼
    ┌─────────────┐
    │ PPO Update  │
    │ - Policy    │
    │ - Value fn  │
    │ - Predictor │
    │ - Encoder*  │ * Phase 2 only
    └─────────────┘
```

## Key Components

### 1. Semantic Encoder (E)

**Role:** Map pixels → semantic features

```python
class SemanticEncoder(nn.Module):
    def __init__(self, model_name="dinov2_vits14"):
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.freeze()  # Phase 1: frozen

    def unfreeze_top_layers(self, n_layers=2):
        # Phase 2: fine-tune last 2 transformer blocks
        ...
```

**Why DINOv2?**
- Pre-trained on 142M images
- Semantic features out-of-the-box
- Patch-level features (14×14 grid)
- Multiple model sizes (21M - 1.1B params)

### 2. Forward Dynamics Model (P)

**Role:** Predict next features from current features + action

```python
class ForwardDynamicsModel(nn.Module):
    def forward(self, features_t, action):
        action_onehot = F.one_hot(action, num_classes=8)
        x = torch.cat([features_t, action_onehot], dim=-1)
        return self.model(x)  # → predicted features_{t+1}

    def compute_intrinsic_reward(self, features_t, action, features_t1):
        predicted = self.forward(features_t, action)
        return torch.sum((predicted - features_t1) ** 2, dim=-1)
```

**Alternative: RND (Random Network Distillation)**
- Fixed random target network
- No action conditioning
- More stable for some tasks

### 3. Navigation Policy (π)

**Role:** Choose actions based on semantic features

```python
class NavigationPolicy(nn.Module):
    def __init__(self, feature_dim, action_dim=8):
        self.actor = ...   # features → action logits
        self.critic = ...  # features → value estimate

    def act(self, features):
        logits, value = self.forward(features)
        action = Categorical(logits=logits).sample()
        return action, log_prob, value
```

**Actions:** 8-connected movement
- 0: RIGHT, 1: DOWN-RIGHT, 2: DOWN, ...
- Extensions: jump actions, scout actions

### 4. Image Navigation Environment

**Role:** MDP formulation for navigation

**State:** `(position, visited_mask, features)`

**Action:** 8-connected movement (or extended with jumps)

**Reward:**
```python
# Prediction error (intrinsic curiosity)
pred_error = predictor.compute_intrinsic_reward(z_t, action, z_t1)

# Multi-step lookahead (exponentially weighted)
lookahead = sum(exp(-λ*d) * prediction_error_at_distance_d
                for d in range(1, horizon))

# Coverage bonus (prevent loops)
coverage_bonus = β / sqrt(visit_count)

total_reward = pred_error + lookahead + coverage_bonus
```

## Two-Phase Training

Critical design choice: **don't train encoder and policy simultaneously!**

### Phase 1: Frozen Encoder (10k episodes)

```python
encoder.freeze()  # All parameters frozen

# Train policy and predictor with stable features
for episode in range(10000):
    rollout = collect_rollout()
    update_policy(rollout)      # PPO
    update_predictor(rollout)   # Forward dynamics
```

**Why freeze?**
- Policy needs stable feature space to learn
- Prevents encoder collapse
- Focuses learning on navigation, not features

### Phase 2: Fine-Tuned Encoder (5k episodes)

```python
encoder.unfreeze_top_layers(n_layers=2)
encoder_lr = 1e-5  # Very small!

for episode in range(5000):
    rollout = collect_rollout()
    update_policy(rollout)
    update_predictor(rollout)
    # Encoder updated via predictor gradients only!
```

**Critical:** Policy uses `features.detach()` to prevent policy loss from corrupting encoder:

```python
# trainer.py:203
with torch.no_grad():
    action, log_prob, value = policy.act(features.detach())
```

Encoder only updated through predictor loss!

## Learned Behaviors

After training, the agent learns to:

1. **Seek semantic boundaries**
   - Car → road, road → sky transitions
   - Object edges have high prediction error

2. **Follow co-occurrence patterns**
   - Cars → roads → lane markers
   - Buildings → windows → sky

3. **Avoid redundant exploration**
   - Coverage bonus prevents loops
   - Visited regions have lower value

4. **Maximize information gain**
   - High-frequency saccades in information-dense regions
   - Quick traversal through uniform areas

## Next Steps

- [Architecture Details](architecture.md) - Deep dive into components
- [Training Guide](training.md) - Best practices and tips
- [Extensions](extensions.md) - Jump/scout actions for long-range exploration
