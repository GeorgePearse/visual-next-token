"""
Extended Action Spaces for RL Navigation

Provides optional extensions to the base 8-connected movement:
- Jump actions: Long-range movement (5-10 pixels)
- Scout actions: Sample distant regions without committing

These are EXTENSIONS to the core strategy, not replacements.
"""

import numpy as np
import torch
import torch.nn as nn


class ExtendedActionSpace:
    """
    Extended action space with jump/scout capabilities.

    Action Types:
    - Base (0-7): 8-connected movement (1 pixel)
    - Jump (8-15): Long-range jumps (5-10 pixels) in 8 directions
    - Scout (16-23): Peek at distant regions without moving
    """

    # Base 8-connected movement (1 pixel step)
    BASE_ACTIONS = {
        0: (0, 1),    # RIGHT
        1: (1, 1),    # DOWN_RIGHT
        2: (1, 0),    # DOWN
        3: (1, -1),   # DOWN_LEFT
        4: (0, -1),   # LEFT
        5: (-1, -1),  # UP_LEFT
        6: (-1, 0),   # UP
        7: (-1, 1),   # UP_RIGHT
    }

    def __init__(self, jump_distance=7, use_jumps=True, use_scouts=False):
        """
        Args:
            jump_distance: Distance for jump actions (pixels)
            use_jumps: Enable jump actions
            use_scouts: Enable scout actions (peek without moving)
        """
        self.jump_distance = jump_distance
        self.use_jumps = use_jumps
        self.use_scouts = use_scouts

        # Build action space
        self.actions = {}
        self.action_types = {}

        # Base actions (0-7)
        for action_id, delta in self.BASE_ACTIONS.items():
            self.actions[action_id] = delta
            self.action_types[action_id] = "base"

        # Jump actions (8-15)
        if use_jumps:
            for i, (dr, dc) in self.BASE_ACTIONS.items():
                action_id = 8 + i
                self.actions[action_id] = (dr * jump_distance, dc * jump_distance)
                self.action_types[action_id] = "jump"

        # Scout actions (16-23)
        if use_scouts:
            for i, (dr, dc) in self.BASE_ACTIONS.items():
                action_id = 16 + i
                self.actions[action_id] = (dr * jump_distance, dc * jump_distance)
                self.action_types[action_id] = "scout"

        self.action_dim = len(self.actions)

    def get_next_position(self, current_pos, action_id, image_shape):
        """
        Get next position given action.

        Args:
            current_pos: (row, col) tuple
            action_id: Action index
            image_shape: (height, width) tuple

        Returns:
            next_pos: (row, col) tuple, clipped to image bounds
            action_type: "base", "jump", or "scout"
        """
        dr, dc = self.actions[action_id]
        next_row = np.clip(current_pos[0] + dr, 0, image_shape[0] - 1)
        next_col = np.clip(current_pos[1] + dc, 0, image_shape[1] - 1)

        return (int(next_row), int(next_col)), self.action_types[action_id]

    def is_scout_action(self, action_id):
        """Check if action is a scout (non-committing peek)."""
        return self.action_types.get(action_id) == "scout"


class HierarchicalPolicy(nn.Module):
    """
    Hierarchical policy for extended action space.

    Uses two-level decision making:
    1. Meta-policy: Choose action type (base vs jump vs scout)
    2. Direction-policy: Choose direction (8 directions)

    This factorization makes learning more efficient.
    """

    def __init__(self, feature_dim, action_space: ExtendedActionSpace, hidden_dim=256):
        """
        Args:
            feature_dim: Dimension of semantic features
            action_space: ExtendedActionSpace instance
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.action_space = action_space
        self.action_dim = action_space.action_dim

        # Determine number of action types
        self.n_types = 1  # base
        if action_space.use_jumps:
            self.n_types += 1
        if action_space.use_scouts:
            self.n_types += 1

        # Shared feature processing
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Meta-policy: Choose action type
        self.meta_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.n_types),
        )

        # Direction policy: Choose direction (8 directions)
        self.direction_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 8),
        )

        # Critic: Value estimate
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        """
        Forward pass through hierarchical policy.

        Args:
            features: Semantic features (batch_size, feature_dim)

        Returns:
            type_logits: Logits for action type (batch_size, n_types)
            direction_logits: Logits for direction (batch_size, 8)
            value: Value estimate (batch_size, 1)
        """
        # Shared processing
        shared_features = self.shared(features)

        # Meta and direction policies
        type_logits = self.meta_policy(shared_features)
        direction_logits = self.direction_policy(shared_features)
        value = self.critic(shared_features)

        return type_logits, direction_logits, value

    def _hierarchical_to_flat_action(self, action_type, direction):
        """
        Convert hierarchical action (type, direction) to flat action ID.

        Args:
            action_type: 0 (base), 1 (jump), or 2 (scout)
            direction: 0-7 (8 directions)

        Returns:
            action_id: Flat action index
        """
        if action_type == 0:
            return direction  # 0-7: base actions
        elif action_type == 1:
            return 8 + direction  # 8-15: jump actions
        else:
            return 16 + direction  # 16-23: scout actions

    def act(self, features, deterministic=False):
        """
        Sample action from hierarchical policy.

        Args:
            features: Semantic features (can be batched or single)
            deterministic: If True, take argmax; if False, sample

        Returns:
            action: Flat action index
            log_prob: Log probability of action
            value: Value estimate
        """
        from torch.distributions import Categorical

        # Ensure batch dimension
        if features.dim() == 1:
            features = features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Get logits
        type_logits, direction_logits, value = self.forward(features)

        # Create distributions
        type_dist = Categorical(logits=type_logits)
        direction_dist = Categorical(logits=direction_logits)

        # Sample or take argmax
        if deterministic:
            action_type = torch.argmax(type_logits, dim=-1)
            direction = torch.argmax(direction_logits, dim=-1)
        else:
            action_type = type_dist.sample()
            direction = direction_dist.sample()

        # Convert to flat action
        batch_size = features.size(0)
        action = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        for i in range(batch_size):
            action[i] = self._hierarchical_to_flat_action(
                action_type[i].item(), direction[i].item()
            )

        # Combined log probability
        log_prob = type_dist.log_prob(action_type) + direction_dist.log_prob(direction)

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
            actions: Flat action indices (batch_size,)

        Returns:
            log_probs: Log probabilities of actions
            values: Value estimates
            entropy: Entropy of action distribution
        """
        # Get logits
        type_logits, direction_logits, values = self.forward(features)

        # Create distributions
        from torch.distributions import Categorical
        type_dist = Categorical(logits=type_logits)
        direction_dist = Categorical(logits=direction_logits)

        # Decompose flat actions into (type, direction)
        action_types = torch.zeros_like(actions)
        directions = torch.zeros_like(actions)

        for i, action in enumerate(actions):
            if action < 8:
                action_types[i] = 0
                directions[i] = action
            elif action < 16:
                action_types[i] = 1
                directions[i] = action - 8
            else:
                action_types[i] = 2
                directions[i] = action - 16

        # Log probabilities
        type_log_probs = type_dist.log_prob(action_types)
        direction_log_probs = direction_dist.log_prob(directions)
        log_probs = type_log_probs + direction_log_probs

        # Entropy (both distributions)
        entropy = type_dist.entropy() + direction_dist.entropy()

        return log_probs, values.squeeze(-1), entropy


class ScoutingRewardModifier:
    """
    Reward modifier for scout actions.

    Scout actions allow peeking at distant regions without committing.
    They receive reduced rewards but provide information for decision making.
    """

    def __init__(self, scout_reward_scale=0.5, scout_penalty=0.01):
        """
        Args:
            scout_reward_scale: Scale factor for scout action rewards (< 1.0)
            scout_penalty: Small penalty for scouting (to prevent abuse)
        """
        self.scout_reward_scale = scout_reward_scale
        self.scout_penalty = scout_penalty

    def modify_reward(self, reward, action_type, is_scout):
        """
        Modify reward based on action type.

        Args:
            reward: Original reward
            action_type: "base", "jump", or "scout"
            is_scout: Boolean, is this a scout action

        Returns:
            modified_reward: Adjusted reward
        """
        if is_scout:
            # Scout actions: reduced reward + small penalty
            return reward * self.scout_reward_scale - self.scout_penalty
        else:
            # Base and jump actions: full reward
            return reward


def test_extended_actions():
    """Test extended action space."""
    print("Testing Extended Action Space...")

    # Create action space
    action_space = ExtendedActionSpace(
        jump_distance=7,
        use_jumps=True,
        use_scouts=False,
    )

    print(f"Total actions: {action_space.action_dim}")
    print(f"Base: 0-7, Jump: 8-15")

    # Test position updates
    current_pos = (100, 100)
    image_shape = (224, 224)

    # Base action (RIGHT)
    next_pos, action_type = action_space.get_next_position(0, current_pos, image_shape)
    assert next_pos == (100, 101)
    assert action_type == "base"
    print(f"Base action 0: {current_pos} -> {next_pos}")

    # Jump action (DOWN_RIGHT jump)
    next_pos, action_type = action_space.get_next_position(9, current_pos, image_shape)
    assert next_pos == (107, 107)
    assert action_type == "jump"
    print(f"Jump action 9: {current_pos} -> {next_pos}")

    print("Extended action space test passed!\n")

    # Test hierarchical policy
    print("Testing Hierarchical Policy...")

    feature_dim = 384
    batch_size = 16

    policy = HierarchicalPolicy(
        feature_dim=feature_dim,
        action_space=action_space,
        hidden_dim=256,
    )

    # Random features
    features = torch.randn(batch_size, feature_dim)

    # Test forward
    type_logits, direction_logits, values = policy(features)
    assert type_logits.shape == (batch_size, 2)  # base + jump
    assert direction_logits.shape == (batch_size, 8)
    assert values.shape == (batch_size, 1)

    # Test action sampling
    action, log_prob, value = policy.act(features[0])
    assert 0 <= action.item() < action_space.action_dim

    # Test batch action sampling
    actions, log_probs, values_out = policy.act(features)
    assert actions.shape == (batch_size,)
    assert log_probs.shape == (batch_size,)

    # Test evaluation
    log_probs_eval, values_eval, entropy = policy.evaluate_actions(features, actions)
    assert log_probs_eval.shape == (batch_size,)
    assert values_eval.shape == (batch_size,)
    assert entropy.shape == (batch_size,)

    print("Hierarchical policy test passed!")
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_extended_actions()
