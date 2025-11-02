# Curiosity-driven Exploration by Self-supervised Prediction (ICM)

## Paper Information

**arXiv ID:** 1705.05363
**Submitted:** May 15, 2017
**Venue:** ICML 2017
**Field:** Computer Science - Machine Learning

## Authors

- Deepak Pathak
- Pulkit Agrawal
- Alexei A. Efros
- Trevor Darrell

## Abstract

The paper addresses exploration in environments with sparse or absent external rewards. The authors propose using curiosity as an intrinsic motivation signal, formulated through an agent's prediction error regarding action consequences in a learned visual feature space. Their approach leverages a self-supervised inverse dynamics model and scales effectively to high-dimensional state spaces like images while avoiding direct pixel prediction.

## Key Contributions

1. **Novel Curiosity Framework:** Formulates intrinsic motivation as prediction error in a learned feature space, enabling exploration without external rewards.

2. **Scalability to Visual Domains:** Successfully handles image-based environments by operating on learned representations rather than raw pixels.

3. **Environmental Relevance:** The method focuses on agent-relevant environmental aspects, ignoring irrelevant visual changes.

4. **Comprehensive Evaluation:** Demonstrates effectiveness across three scenarios:
   - Sparse reward environments requiring fewer interactions
   - Unrewarded exploration showing efficient behavior
   - Generalization to new game levels leveraging prior knowledge

## Resources

- **arXiv:** https://arxiv.org/abs/1705.05363
- **Website:** pathak22.github.io/noreward-rl/
- **PDF:** Available on arXiv
- **Code & Demo:** Available via project website

## Relevance to Our Implementation

This paper forms the theoretical foundation for our **ForwardDynamicsModel** class. The ICM (Intrinsic Curiosity Module) approach:

- **Forward Model:** Predicts next features from current features + action
  ```python
  predicted_features = forward_model(features_t, action)
  intrinsic_reward = ||predicted_features - features_t1||²
  ```

- **Feature Learning:** Uses inverse dynamics to learn agent-relevant features (solves "car color problem")
- **Prediction Error as Reward:** High prediction error = novel/surprising state = exploration bonus

In our code (techniques/rl_navigation/forward_dynamics.py:122-133):
```python
class ForwardDynamicsModel:
    def compute_intrinsic_reward(self, features_t, action, features_t1):
        predicted_features = self.forward(features_t, action)
        return torch.sum((predicted_features - features_t1) ** 2, dim=-1)
```

Our implementation differs by:
1. Using pre-trained DINOv2 features instead of learning features from scratch
2. Two-phase training: frozen encoder → fine-tuned encoder
3. Exponential distance weighting for multi-step lookahead rewards
