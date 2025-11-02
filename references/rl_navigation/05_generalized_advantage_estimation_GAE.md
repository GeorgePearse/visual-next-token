# High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)

## Paper Information

**arXiv ID:** 1506.02438
**Submitted:** June 8, 2015 (v1)
**Last Revised:** October 20, 2018 (v6)
**Subject Areas:** Machine Learning (cs.LG), Robotics (cs.RO), Systems and Control (eess.SY)

## Authors

- John Schulman
- Philipp Moritz
- Sergey Levine
- Michael Jordan
- Pieter Abbeel

## Abstract

The paper introduces an approach to policy gradient methods in reinforcement learning that addresses two key challenges: high sample complexity and training instability. The authors employ value functions to reduce variance in policy gradient estimates while introducing some bias through "an exponentially-weighted estimator of the advantage function that is analogous to TD(lambda)." For stability, they use trust region optimization for both policy and value function neural networks.

The method demonstrates strong performance on complex 3D locomotion tasks, including bipedal and quadrupedal robot locomotion, with policies learned directly from raw kinematics to joint torques—requiring simulated experience equivalent to 1-2 weeks of real time for 3D bipeds.

## Key Contributions

- **Variance reduction technique:** Leverages value functions with a generalized advantage estimation method to improve sample efficiency
- **Stable optimization:** Applies trust region methods to both policy and value functions
- **End-to-end learning:** Direct mapping from kinematics to control without hand-crafted representations
- **Empirical validation:** Successful learning of complex locomotion behaviors in high-dimensional continuous control domains

## Resources

- **arXiv:** https://arxiv.org/abs/1506.02438

## Relevance to Our Implementation

GAE provides the **advantage estimation** used in our PPO implementation to compute policy gradients with reduced variance. Implementation in techniques/rl_navigation/policy.py:190-223.

### The Bias-Variance Tradeoff

**Problem:** Policy gradient estimation
- High variance → slow learning, unstable
- High bias → incorrect gradients, poor convergence

**Solution:** GAE with lambda parameter to balance bias-variance

### GAE Formula

The advantage function estimates "how much better is action a than average at state s":

```
A(s_t, a_t) = Q(s_t, a_t) - V(s_t)
```

GAE uses an exponentially-weighted average of n-step advantages:

```
GAE(λ) = (1-λ) * [A^(1) + λ*A^(2) + λ²*A^(3) + ...]
```

where:
```
A^(n) = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n*V(s_{t+n}) - V(s_t)
```

### Implementation (policy.py:190-223)

```python
def compute_gae(self, rewards, values, dones):
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards (T,)
        values: Value estimates (T+1,) - includes next value
        dones: Done flags (T,)

    Returns:
        advantages: GAE advantages (T,)
        returns: Discounted returns (T,)
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0

    # Compute advantages backwards in time
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = values[t + 1]
        else:
            next_value = values[t + 1]

        # TD error: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

        # GAE: A_t = δ_t + γ*λ*A_{t+1}
        advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
        last_advantage = advantages[t]

    # Returns = advantages + values
    returns = advantages + values[:-1]

    return advantages, returns
```

### GAE Hyperparameter: Lambda (λ)

Controls the bias-variance tradeoff:

| λ value | Behavior | Variance | Bias |
|---------|----------|----------|------|
| λ = 0 | 1-step TD (bootstrapping) | Low | High |
| λ = 1 | Monte Carlo (full returns) | High | Low |
| λ = 0.95 | **Balanced (our default)** | Medium | Medium |

In our config (config.py:26):
```python
gae_lambda: float = 0.95  # GAE lambda
```

### Why GAE for Image Navigation?

1. **Long Episodes:** Our episodes can be 500+ steps
   - Monte Carlo (λ=1) would have huge variance
   - Pure TD (λ=0) would have high bias

2. **Sparse Rewards:** Intrinsic rewards are noisy
   - GAE smooths reward signal
   - Reduces impact of prediction noise

3. **Credit Assignment:** Multi-step lookahead rewards
   - GAE helps assign credit across time
   - Important for exponentially-weighted future prediction

### Integration with PPO (policy.py:251-256)

```python
# Compute advantages with GAE
advantages, returns = self.compute_gae(rewards, values, dones)

# Normalize advantages (reduce variance)
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Use in PPO loss
policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
value_loss = F.mse_loss(values_pred, returns)
```

### Empirical Results

The paper showed GAE significantly improves:
- Sample efficiency (fewer episodes to convergence)
- Training stability (smoother learning curves)
- Final performance (higher rewards)

These benefits directly apply to our image navigation task, where we need stable learning with limited interaction budget (15k episodes total).
