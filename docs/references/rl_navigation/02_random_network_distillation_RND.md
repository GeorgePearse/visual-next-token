# Exploration by Random Network Distillation (RND)

## Paper Information

**ArXiv ID:** 1810.12894
**Submission Date:** October 30, 2018
**Field:** Computer Science > Machine Learning

## Authors
- Yuri Burda
- Harrison Edwards
- Amos Storkey
- Oleg Klimov

## Abstract

The paper introduces "an exploration bonus for deep reinforcement learning methods that is easy to implement and adds minimal overhead to the computation performed." The mechanism leverages prediction error from a neural network trained on fixed random features as an intrinsic reward signal.

## Key Contributions

1. **Random Network Distillation (RND) Bonus:** A novel exploration mechanism based on measuring how well a trainable predictor can estimate features from a fixed random network.

2. **Flexible Reward Combination:** A method for adaptively merging intrinsic (exploration) and extrinsic (task) rewards during training.

3. **State-of-the-Art Results:** Achieved breakthrough performance on Montezuma's Revenge, "the first method that achieves better than average human performance on this game without using demonstrations or having access to the underlying state," occasionally completing the first level.

4. **Hard Exploration Benchmarks:** Demonstrated significant improvements across multiple challenging Atari games known for sparse reward signals.

## Resources

- **arXiv:** https://arxiv.org/abs/1810.12894
- **DOI:** https://doi.org/10.48550/arXiv.1810.12894
- **Subject Categories:** Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Machine Learning (stat.ML)

## Relevance to Our Implementation

RND provides an **alternative intrinsic motivation mechanism** to ICM. Our implementation includes both options (see techniques/rl_navigation/forward_dynamics.py:143-275):

### RND Architecture

```python
class RNDIntrinsicMotivation:
    def __init__(self, feature_dim, hidden_dim=512):
        # Fixed random target network (never updated)
        self.target_net = self._build_network(feature_dim, hidden_dim)
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Trainable predictor network
        self.predictor_net = self._build_network(feature_dim, hidden_dim)

    def compute_intrinsic_reward(self, features):
        with torch.no_grad():
            target = self.target_net(features)
        prediction = self.predictor_net(features)
        return torch.sum((prediction - target) ** 2, dim=-1)
```

### RND vs ICM Comparison

| Aspect | RND | ICM (Forward Dynamics) |
|--------|-----|------------------------|
| **Prediction Target** | Fixed random features | Next state features |
| **Requires Actions** | No | Yes |
| **Novelty Signal** | Feature novelty | Transition novelty |
| **Stability** | More stable (fixed target) | Requires careful tuning |
| **Agent Relevance** | Less focused | More action-relevant |

### Usage in Our Code

```python
# Create RND-based trainer
trainer = RLTrainer(
    image=image,
    use_rnd=True,  # Use RND instead of forward dynamics
    ...
)

# Or use config
from techniques.rl_navigation.config import get_config
config = get_config("rnd")
```

RND is particularly useful when:
- Action-conditioned prediction is difficult
- You want more stable intrinsic rewards
- Feature-space novelty is the primary exploration signal
