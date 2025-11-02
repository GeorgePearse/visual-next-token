"""
Image Navigation Environment

MDP formulation for navigating through images to maximize prediction error
of future semantic features.

State: (position, visited_mask, local_features)
Action: 8-connected movement directions
Reward: Exponentially weighted prediction error of future positions
"""

import numpy as np
import torch


class ImageNavigationEnv:
    """
    RL environment for learning to navigate images.

    The agent learns to move through an image to maximize its ability to
    predict future semantic features - leading it to explore information-dense
    regions with high semantic co-occurrence (car → road + sky).
    """

    # 8-connected movement directions
    ACTIONS = {
        0: (0, 1),    # RIGHT
        1: (1, 1),    # DOWN_RIGHT
        2: (1, 0),    # DOWN
        3: (1, -1),   # DOWN_LEFT
        4: (0, -1),   # LEFT
        5: (-1, -1),  # UP_LEFT
        6: (-1, 0),   # UP
        7: (-1, 1),   # UP_RIGHT
    }

    ACTION_NAMES = ["RIGHT", "DOWN_RIGHT", "DOWN", "DOWN_LEFT",
                    "LEFT", "UP_LEFT", "UP", "UP_RIGHT"]

    def __init__(
        self,
        image,
        encoder,
        predictor,
        max_steps=500,
        reward_horizon=10,
        reward_lambda=0.1,
        coverage_bonus_weight=0.1,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            image: RGB image tensor (C, H, W) or numpy array (H, W, C)
            encoder: SemanticEncoder for extracting features
            predictor: ForwardDynamicsModel for computing prediction error
            max_steps: Maximum steps per episode
            reward_horizon: How far ahead to look for prediction error
            reward_lambda: Decay rate for exponential distance weighting
            coverage_bonus_weight: Weight for coverage/novelty bonus
            device: Device for computation
        """
        # Convert image to tensor if needed
        if isinstance(image, np.ndarray):
            # Assume (H, W, C) numpy array
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        self.image = image.to(device)
        self.encoder = encoder
        self.predictor = predictor
        self.max_steps = max_steps
        self.reward_horizon = reward_horizon
        self.reward_lambda = reward_lambda
        self.coverage_bonus_weight = coverage_bonus_weight
        self.device = device

        # Image dimensions
        self.C, self.H, self.W = self.image.shape

        # Precompute all patch features for efficiency
        print("Precomputing semantic features...")
        self.patch_features = encoder.precompute_patch_features(self.image)
        self.num_patches_h, self.num_patches_w, self.feature_dim = self.patch_features.shape
        print(f"Patch grid: {self.num_patches_h} x {self.num_patches_w}")

        # Episode state
        self.current_pos = None
        self.visited = None
        self.visit_count = None
        self.step_count = 0
        self.path_history = []

    def reset(self, start_pos=None):
        """
        Reset environment for new episode.

        Args:
            start_pos: Optional (row, col) starting position.
                      If None, randomly samples position.

        Returns:
            state: Initial state dictionary
        """
        # Random starting position if not specified
        if start_pos is None:
            start_pos = (
                np.random.randint(0, self.H),
                np.random.randint(0, self.W),
            )

        self.current_pos = start_pos
        self.step_count = 0
        self.path_history = [start_pos]

        # Initialize visited mask and visit counts
        self.visited = np.zeros((self.H, self.W), dtype=bool)
        self.visit_count = np.zeros((self.H, self.W), dtype=int)

        self.visited[start_pos] = True
        self.visit_count[start_pos] = 1

        return self._get_state()

    def _get_state(self):
        """
        Get current state representation.

        Returns:
            state: Dictionary with:
                - features: Semantic features at current position
                - position: (row, col) tuple
                - visited: Binary mask of visited positions
                - visit_count: Count of visits to each position
        """
        # Convert pixel position to patch position
        row, col = self.current_pos
        patch_row = min(row // self.encoder.patch_size, self.num_patches_h - 1)
        patch_col = min(col // self.encoder.patch_size, self.num_patches_w - 1)

        # Get semantic features at current patch
        features = self.patch_features[patch_row, patch_col]

        return {
            "features": features,
            "position": self.current_pos,
            "visited": self.visited.copy(),
            "visit_count": self.visit_count.copy(),
        }

    def step(self, action):
        """
        Execute action and return next state, reward, done.

        Args:
            action: Integer in [0, 7] representing direction

        Returns:
            next_state: State dictionary
            reward: Intrinsic reward (prediction error + bonuses)
            done: Whether episode is finished
            info: Additional information dictionary
        """
        # Get movement delta
        dr, dc = self.ACTIONS[action]

        # Compute next position
        next_row = self.current_pos[0] + dr
        next_col = self.current_pos[1] + dc

        # Check bounds
        if not (0 <= next_row < self.H and 0 <= next_col < self.W):
            # Hit boundary - terminate episode
            return self._get_state(), -1.0, True, {"termination": "boundary"}

        # Check if stuck (all neighbors visited)
        if self._is_stuck():
            return self._get_state(), 0.0, True, {"termination": "stuck"}

        # Update position
        prev_pos = self.current_pos
        self.current_pos = (next_row, next_col)
        self.visited[next_row, next_col] = True
        self.visit_count[next_row, next_col] += 1
        self.path_history.append(self.current_pos)
        self.step_count += 1

        # Compute reward
        reward, reward_info = self._compute_reward(prev_pos, action, self.current_pos)

        # Check termination
        done = self.step_count >= self.max_steps

        info = {
            "termination": "max_steps" if done else None,
            "reward_breakdown": reward_info,
        }

        return self._get_state(), reward, done, info

    def _compute_reward(self, prev_pos, action, current_pos):
        """
        Compute intrinsic reward based on prediction error.

        Reward = Σ exp(-λ·d) · ||predicted_features - actual_features||²
                + coverage_bonus

        Args:
            prev_pos: Previous (row, col) position
            action: Action taken
            current_pos: New (row, col) position

        Returns:
            reward: Total reward
            reward_info: Dictionary with reward breakdown
        """
        # Get features at previous and current position
        prev_patch = self._pos_to_patch(prev_pos)
        curr_patch = self._pos_to_patch(current_pos)

        prev_features = self.patch_features[prev_patch[0], prev_patch[1]]

        # Predict current features from previous
        action_tensor = torch.tensor([action], device=self.device)
        predicted_features = self.predictor(
            prev_features.unsqueeze(0),
            action_tensor
        ).squeeze(0)

        # Prediction error = intrinsic reward
        curr_features = self.patch_features[curr_patch[0], curr_patch[1]]
        prediction_error = torch.sum((predicted_features - curr_features) ** 2).item()

        # Optional: Look ahead to future positions (exponentially weighted)
        lookahead_reward = 0.0
        if self.reward_horizon > 1:
            lookahead_reward = self._compute_lookahead_reward(current_pos)

        # Coverage bonus: Encourage visiting new regions
        coverage_bonus = self.coverage_bonus_weight / np.sqrt(
            self.visit_count[current_pos] + 1e-8
        )

        # Total reward
        total_reward = prediction_error + lookahead_reward + coverage_bonus

        reward_info = {
            "prediction_error": prediction_error,
            "lookahead_reward": lookahead_reward,
            "coverage_bonus": coverage_bonus,
            "total": total_reward,
        }

        return total_reward, reward_info

    def _compute_lookahead_reward(self, current_pos):
        """
        Compute exponentially weighted prediction error for future positions.

        Args:
            current_pos: Current (row, col) position

        Returns:
            lookahead_reward: Weighted sum of future prediction errors
        """
        total_reward = 0.0
        curr_patch = self._pos_to_patch(current_pos)
        curr_features = self.patch_features[curr_patch[0], curr_patch[1]]

        # Look ahead in all 8 directions
        for distance in range(1, self.reward_horizon):
            weight = np.exp(-self.reward_lambda * distance)

            for action_id, (dr, dc) in self.ACTIONS.items():
                # Future position at this distance
                future_row = current_pos[0] + dr * distance
                future_col = current_pos[1] + dc * distance

                # Check bounds
                if not (0 <= future_row < self.H and 0 <= future_col < self.W):
                    continue

                # Get future features
                future_patch = self._pos_to_patch((future_row, future_col))
                future_features = self.patch_features[future_patch[0], future_patch[1]]

                # Predict future features
                action_tensor = torch.tensor([action_id], device=self.device)
                predicted_future = self.predictor(
                    curr_features.unsqueeze(0),
                    action_tensor
                ).squeeze(0)

                # Prediction error
                error = torch.sum((predicted_future - future_features) ** 2).item()

                total_reward += weight * error

        return total_reward

    def _pos_to_patch(self, pos):
        """Convert pixel position to patch position."""
        row, col = pos
        patch_row = min(row // self.encoder.patch_size, self.num_patches_h - 1)
        patch_col = min(col // self.encoder.patch_size, self.num_patches_w - 1)
        return (patch_row, patch_col)

    def _is_stuck(self):
        """Check if agent is stuck (all neighbors visited)."""
        row, col = self.current_pos

        for dr, dc in self.ACTIONS.values():
            next_row, next_col = row + dr, col + dc

            if 0 <= next_row < self.H and 0 <= next_col < self.W:
                if not self.visited[next_row, next_col]:
                    return False

        return True

    def render(self):
        """
        Visualize current state (for debugging).

        Returns:
            visualization: RGB image with path overlaid
        """
        import cv2

        # Convert image to numpy for visualization
        vis_image = (self.image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()

        # Draw visited regions (semi-transparent)
        visited_mask = np.stack([self.visited] * 3, axis=-1)
        vis_image[visited_mask] = (vis_image[visited_mask] * 0.7 + np.array([0, 255, 0]) * 0.3).astype(np.uint8)

        # Draw path
        for i in range(len(self.path_history) - 1):
            pt1 = (self.path_history[i][1], self.path_history[i][0])  # (col, row)
            pt2 = (self.path_history[i + 1][1], self.path_history[i + 1][0])
            cv2.line(vis_image, pt1, pt2, (255, 0, 0), 2)

        # Mark current position
        curr_pt = (self.current_pos[1], self.current_pos[0])
        cv2.circle(vis_image, curr_pt, 5, (0, 0, 255), -1)

        return vis_image

    def get_statistics(self):
        """
        Get episode statistics.

        Returns:
            stats: Dictionary with coverage, path length, etc.
        """
        return {
            "step_count": self.step_count,
            "coverage": np.sum(self.visited) / (self.H * self.W),
            "unique_positions": len(self.path_history),
            "path_length": self.step_count,
            "revisit_rate": 1.0 - len(set(self.path_history)) / len(self.path_history),
        }
