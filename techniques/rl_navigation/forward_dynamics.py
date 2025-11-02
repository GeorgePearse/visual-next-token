"""
Forward Dynamics Model & Intrinsic Motivation

Two approaches for generating intrinsic signals:
1. Forward Dynamics Model (ICM): Predict next features from current + action
2. Random Network Distillation (RND): Predict fixed random network output

These models compute prediction ERROR, which is then NEGATED in the environment
to create accuracy-based rewards. The agent maximizes rolling-window prediction
accuracy (low error = high reward), seeking semantically coherent paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardDynamicsModel(nn.Module):
    """
    Forward dynamics model for Intrinsic Curiosity Module (ICM).

    Predicts next semantic features given current features and action.

    NOTE: This model outputs PREDICTION ERROR, which the environment then
    NEGATES to create accuracy rewards (low error = high reward). The agent
    seeks paths with high rolling-window prediction accuracy.

    Architecture:
        Input: concat(features_t, action_onehot) → MLP → predicted_features_t+1
    """

    def __init__(self, feature_dim, action_dim=8, hidden_dim=512):
        """
        Args:
            feature_dim: Dimension of semantic features from encoder
            action_dim: Number of possible actions (8 for 8-connected movement)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim

        # MLP: (features + action) → predicted next features
        self.model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, features_t, action):
        """
        Predict next features.

        Args:
            features_t: Current semantic features (batch_size, feature_dim)
            action: Action taken (batch_size,) - integer indices

        Returns:
            predicted_features: Predicted next features (batch_size, feature_dim)
        """
        # Convert action to one-hot
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()

        # Concatenate features and action
        x = torch.cat([features_t, action_onehot], dim=-1)

        # Predict next features
        predicted_features = self.model(x)

        return predicted_features

    def compute_intrinsic_reward(self, features_t, action, features_t1):
        """
        Compute prediction error for intrinsic motivation.

        NOTE: Returns PREDICTION ERROR, not reward. The environment negates
        this to create accuracy rewards (reward = -error).

        Args:
            features_t: Current features (batch_size, feature_dim)
            action: Action taken (batch_size,)
            features_t1: Actual next features (batch_size, feature_dim)

        Returns:
            prediction_error: Squared prediction error (batch_size,)
                             Will be negated by environment for reward
        """
        # Predict next features
        predicted_features = self.forward(features_t, action)

        # Compute MSE prediction error
        prediction_error = torch.sum((predicted_features - features_t1) ** 2, dim=-1)

        return prediction_error

    def update(self, features_t, action, features_t1, optimizer):
        """
        Update forward model to minimize prediction error.

        Args:
            features_t: Current features
            action: Actions taken
            features_t1: Actual next features
            optimizer: Optimizer for model parameters

        Returns:
            loss: Prediction loss
        """
        # Predict
        predicted_features = self.forward(features_t, action)

        # Loss: minimize prediction error
        loss = F.mse_loss(predicted_features, features_t1)

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


class RNDIntrinsicMotivation(nn.Module):
    """
    Random Network Distillation (RND) for intrinsic motivation.

    More stable than forward dynamics - avoids "noisy TV" problem.
    Uses fixed random target network that predictor learns to match.

    NOTE: Returns prediction ERROR which is negated by the environment to
    create accuracy rewards. Novel states have high error → low reward,
    familiar states have low error → high reward. This encourages finding
    semantically coherent paths.

    Reference: "Exploration by Random Network Distillation" (Burda et al., 2018)
    """

    def __init__(self, feature_dim, hidden_dim=512, output_dim=128):
        """
        Args:
            feature_dim: Dimension of semantic features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension for target/predictor
        """
        super().__init__()

        self.feature_dim = feature_dim

        # Target network (fixed, random initialization)
        self.target_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Freeze target network
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Predictor network (learns to match target)
        self.predictor_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        """
        Compute target and predictor outputs.

        Args:
            features: Semantic features (batch_size, feature_dim)

        Returns:
            target_output: Fixed random network output
            predictor_output: Predictor network output
        """
        with torch.no_grad():
            target_output = self.target_net(features)

        predictor_output = self.predictor_net(features)

        return target_output, predictor_output

    def compute_intrinsic_reward(self, features):
        """
        Compute prediction error for intrinsic motivation.

        NOTE: Returns PREDICTION ERROR, not reward. The environment negates
        this to create accuracy rewards (reward = -error).

        After negation by environment:
        - Novel states → large error → LOW reward (penalized)
        - Familiar states → small error → HIGH reward (encouraged)

        This encourages semantically coherent paths where prediction is easy.

        Args:
            features: Semantic features (batch_size, feature_dim)

        Returns:
            prediction_error: Squared prediction error (batch_size,)
                             Will be negated by environment for reward
        """
        target_output, predictor_output = self.forward(features)

        # MSE between target and predictor
        prediction_error = torch.sum((target_output - predictor_output) ** 2, dim=-1)

        return prediction_error

    def update(self, features, optimizer):
        """
        Update predictor to match target network.

        Args:
            features: Semantic features
            optimizer: Optimizer for predictor parameters

        Returns:
            loss: Prediction loss
        """
        target_output, predictor_output = self.forward(features)

        # Loss: minimize distance to target
        loss = F.mse_loss(predictor_output, target_output.detach())

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


class CombinedIntrinsicMotivation(nn.Module):
    """
    Combine both forward dynamics and RND for robust exploration.

    Uses weighted sum of both intrinsic rewards:
    - Forward dynamics: Captures action-dependent dynamics
    - RND: Pure novelty detection, action-independent

    This combination can be more robust than either alone.
    """

    def __init__(
        self,
        feature_dim,
        action_dim=8,
        forward_weight=0.5,
        rnd_weight=0.5,
        hidden_dim=512,
    ):
        """
        Args:
            feature_dim: Dimension of semantic features
            action_dim: Number of actions
            forward_weight: Weight for forward dynamics reward
            rnd_weight: Weight for RND reward
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.forward_model = ForwardDynamicsModel(feature_dim, action_dim, hidden_dim)
        self.rnd_model = RNDIntrinsicMotivation(feature_dim, hidden_dim)

        self.forward_weight = forward_weight
        self.rnd_weight = rnd_weight

    def compute_intrinsic_reward(self, features_t, action, features_t1):
        """
        Compute combined intrinsic reward.

        Args:
            features_t: Current features
            action: Action taken
            features_t1: Next features

        Returns:
            reward: Combined intrinsic reward
        """
        # Forward dynamics reward (action-dependent)
        forward_reward = self.forward_model.compute_intrinsic_reward(
            features_t, action, features_t1
        )

        # RND reward (pure novelty, action-independent)
        rnd_reward = self.rnd_model.compute_intrinsic_reward(features_t1)

        # Weighted combination
        total_reward = (
            self.forward_weight * forward_reward + self.rnd_weight * rnd_reward
        )

        return total_reward

    def update(self, features_t, action, features_t1, forward_optimizer, rnd_optimizer):
        """
        Update both models.

        Returns:
            losses: Dictionary with both losses
        """
        forward_loss = self.forward_model.update(
            features_t, action, features_t1, forward_optimizer
        )

        rnd_loss = self.rnd_model.update(features_t1, rnd_optimizer)

        return {"forward_loss": forward_loss, "rnd_loss": rnd_loss}


def test_forward_dynamics():
    """Test forward dynamics model."""
    feature_dim = 384  # DINOv2 ViT-B feature dim
    batch_size = 32

    # Create model
    model = ForwardDynamicsModel(feature_dim, action_dim=8)

    # Random features and actions
    features_t = torch.randn(batch_size, feature_dim)
    actions = torch.randint(0, 8, (batch_size,))
    features_t1 = torch.randn(batch_size, feature_dim)

    # Test forward pass
    predicted = model(features_t, actions)
    assert predicted.shape == (batch_size, feature_dim)

    # Test reward computation
    reward = model.compute_intrinsic_reward(features_t, actions, features_t1)
    assert reward.shape == (batch_size,)

    print("Forward dynamics test passed!")


def test_rnd():
    """Test RND model."""
    feature_dim = 384
    batch_size = 32

    # Create model
    model = RNDIntrinsicMotivation(feature_dim)

    # Random features
    features = torch.randn(batch_size, feature_dim)

    # Test forward pass
    target, predictor = model(features)
    assert target.shape == predictor.shape

    # Test reward
    reward = model.compute_intrinsic_reward(features)
    assert reward.shape == (batch_size,)

    print("RND test passed!")


if __name__ == "__main__":
    test_forward_dynamics()
    test_rnd()
    print("All tests passed!")
