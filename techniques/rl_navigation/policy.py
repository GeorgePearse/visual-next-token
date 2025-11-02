"""
Navigation Policy Network

PPO-based actor-critic policy for learning to navigate images.

Actor: Semantic features → action probabilities (8 directions)
Critic: Semantic features → value estimate (expected future reward)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class NavigationPolicy(nn.Module):
    """
    PPO-style actor-critic policy for image navigation.

    Takes semantic features and outputs:
    - Action distribution (which direction to move)
    - Value estimate (expected cumulative reward)
    """

    def __init__(self, feature_dim, action_dim=8, hidden_dim=256):
        """
        Args:
            feature_dim: Dimension of semantic features from encoder
            action_dim: Number of actions (8 for 8-connected movement)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim

        # Shared feature processing
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Actor head: features → action logits
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Critic head: features → value estimate
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        """
        Forward pass through policy.

        Args:
            features: Semantic features (batch_size, feature_dim)

        Returns:
            action_logits: Logits for action distribution (batch_size, action_dim)
            value: Value estimate (batch_size, 1)
        """
        # Shared processing
        shared_features = self.shared(features)

        # Actor and critic outputs
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)

        return action_logits, value

    def act(self, features, deterministic=False):
        """
        Sample action from policy.

        Args:
            features: Semantic features (can be batched or single)
            deterministic: If True, take argmax; if False, sample

        Returns:
            action: Sampled action index
            log_prob: Log probability of action
            value: Value estimate
        """
        # Ensure batch dimension
        if features.dim() == 1:
            features = features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Get action logits and value
        action_logits, value = self.forward(features)

        # Create action distribution
        action_dist = Categorical(logits=action_logits)

        # Sample or take argmax
        if deterministic:
            action = torch.argmax(action_logits, dim=-1)
        else:
            action = action_dist.sample()

        # Log probability
        log_prob = action_dist.log_prob(action)

        if squeeze_output:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)

        return action, log_prob, value

    def evaluate_actions(self, features, actions):
        """
        Evaluate actions (for PPO update).

        Args:
            features: Semantic features (batch_size, feature_dim)
            actions: Actions taken (batch_size,)

        Returns:
            log_probs: Log probabilities of actions
            values: Value estimates
            entropy: Entropy of action distribution
        """
        # Get action logits and values
        action_logits, values = self.forward(features)

        # Create distribution
        action_dist = Categorical(logits=action_logits)

        # Log probabilities of actual actions
        log_probs = action_dist.log_prob(actions)

        # Entropy (for exploration bonus)
        entropy = action_dist.entropy()

        return log_probs, values.squeeze(-1), entropy


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) trainer.

    Handles:
    - Collecting rollouts
    - Computing advantages (GAE)
    - Policy updates with clipped objective
    - Value function updates
    """

    def __init__(
        self,
        policy,
        optimizer,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
    ):
        """
        Args:
            policy: NavigationPolicy
            optimizer: Optimizer for policy parameters
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Weight for value loss
            entropy_coef: Weight for entropy bonus
            max_grad_norm: Max gradient norm for clipping
        """
        self.policy = policy
        self.optimizer = optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

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

            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            # GAE
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        # Returns = advantages + values
        returns = advantages + values[:-1]

        return advantages, returns

    def update(self, rollout_buffer, n_epochs=4, batch_size=64):
        """
        Update policy with PPO.

        Args:
            rollout_buffer: Dictionary with:
                - features: (T, feature_dim)
                - actions: (T,)
                - old_log_probs: (T,)
                - rewards: (T,)
                - values: (T+1,) - includes next value
                - dones: (T,)
            n_epochs: Number of PPO epochs
            batch_size: Minibatch size

        Returns:
            metrics: Dictionary with training metrics
        """
        # Extract from buffer
        features = rollout_buffer["features"]
        actions = rollout_buffer["actions"]
        old_log_probs = rollout_buffer["old_log_probs"]
        rewards = rollout_buffer["rewards"]
        values = rollout_buffer["values"]
        dones = rollout_buffer["dones"]

        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training data
        dataset_size = len(features)

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }

        n_updates = 0

        # PPO epochs
        for epoch in range(n_epochs):
            # Shuffle indices
            indices = torch.randperm(dataset_size)

            # Minibatch updates
            for start_idx in range(0, dataset_size, batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]

                # Minibatch data
                batch_features = features[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate current policy
                log_probs, values_pred, entropy = self.policy.evaluate_actions(
                    batch_features, batch_actions
                )

                # Ratio for PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values_pred, batch_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.mean().item()
                metrics["total_loss"] += loss.item()

                # Approximate KL and clip fraction
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

                metrics["approx_kl"] += approx_kl
                metrics["clip_fraction"] += clip_fraction

                n_updates += 1

        # Average metrics
        for key in metrics:
            metrics[key] /= n_updates

        return metrics


def test_policy():
    """Test navigation policy."""
    feature_dim = 384
    batch_size = 32

    # Create policy
    policy = NavigationPolicy(feature_dim, action_dim=8)

    # Random features
    features = torch.randn(batch_size, feature_dim)

    # Test forward
    action_logits, values = policy(features)
    assert action_logits.shape == (batch_size, 8)
    assert values.shape == (batch_size, 1)

    # Test action sampling
    action, log_prob, value = policy.act(features[0])
    assert action.shape == ()
    assert log_prob.shape == ()

    # Test evaluation
    actions = torch.randint(0, 8, (batch_size,))
    log_probs, values, entropy = policy.evaluate_actions(features, actions)
    assert log_probs.shape == (batch_size,)
    assert values.shape == (batch_size,)
    assert entropy.shape == (batch_size,)

    print("Policy test passed!")


if __name__ == "__main__":
    test_policy()
