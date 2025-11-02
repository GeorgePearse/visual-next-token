# Proximal Policy Optimization Algorithms (PPO)

## Paper Information

**arXiv ID:** 1707.06347
**Category:** Computer Science > Machine Learning (cs.LG)
**Submitted:** July 20, 2017
**Last Revised:** August 28, 2017 (v2)

## Authors

- John Schulman
- Filip Wolski
- Prafulla Dhariwal
- Alec Radford
- Oleg Klimov

## Abstract

The authors introduce a novel approach to reinforcement learning through "a new family of policy gradient methods" that alternates between environment interaction and objective function optimization. Unlike conventional policy gradient techniques that perform single updates per sample, this work presents a method enabling "multiple epochs of minibatch updates." The resulting algorithm, termed PPO, incorporates advantages from trust region methods while maintaining simplicity and improved empirical sample efficiency. Testing encompasses robotic simulation and Atari games, demonstrating superior performance relative to comparable online policy approaches.

## Key Contributions

1. **Novel Objective Function:** Enables multiple minibatch optimization epochs per data collection phase, improving computational efficiency

2. **Simplified Implementation:** Maintains benefits of trust region policy optimization (TRPO) while reducing implementation complexity

3. **Enhanced Performance:** Empirical results show improved sample complexity and wall-time efficiency across benchmark tasks

4. **Versatile Application:** Successfully applies to diverse domains including simulated robotics and video game playing

## Resources

- **arXiv:** https://arxiv.org/abs/1707.06347
- **DOI:** https://doi.org/10.48550/arXiv.1707.06347
- **OpenAI Blog:** https://openai.com/blog/openai-baselines-ppo/

## Relevance to Our Implementation

PPO is the **core policy learning algorithm** in our navigation system. Implementation in techniques/rl_navigation/policy.py:148-341.

### PPO Clipped Objective

The key innovation is the clipped surrogate objective that prevents too-large policy updates:

```python
class PPOTrainer:
    def update(self, rollout_buffer, n_epochs=4, batch_size=64):
        # Compute probability ratio
        ratio = torch.exp(log_probs - batch_old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * batch_advantages
        surr2 = torch.clamp(
            ratio,
            1 - self.clip_epsilon,  # Typically 0.2
            1 + self.clip_epsilon
        ) * batch_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
```

### Complete PPO Update

Our implementation includes all PPO components:

1. **Policy Loss** (clipped objective)
2. **Value Loss** (MSE with returns)
3. **Entropy Bonus** (exploration)

```python
# Total loss (policy.py:308-312)
loss = (
    policy_loss
    + self.value_loss_coef * value_loss
    + self.entropy_coef * entropy_loss
)
```

### PPO Hyperparameters (config.py)

```python
@dataclass
class RLConfig:
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda
    clip_epsilon: float = 0.2     # PPO clipping

    rollout_steps: int = 2048     # Steps per rollout
    ppo_epochs: int = 4           # PPO update epochs
    ppo_batch_size: int = 64
    max_grad_norm: float = 0.5
```

### Why PPO for Image Navigation?

PPO is ideal for our task because:

1. **Sample Efficiency:** Reuses experience through multiple epochs
2. **Stability:** Clipped objective prevents catastrophic policy collapse
3. **Simplicity:** No complex second-order optimization (vs TRPO)
4. **Works with Continuous Features:** Handles high-dimensional DINOv2 features (384-1024 dims)

### Training Loop (trainer.py:295-343)

```python
def train(self):
    # Phase 1: Frozen encoder
    for episode in range(self.phase1_episodes):
        rollout = self.collect_rollout()  # Gather experiences
        ppo_metrics = self.ppo_trainer.update(rollout)  # PPO update

    # Phase 2: Fine-tuned encoder
    self.encoder.unfreeze_top_layers(n_layers=2)
    for episode in range(self.phase2_episodes):
        ...
```
