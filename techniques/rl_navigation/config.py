"""
Configuration for RL-based Image Navigation

Centralized hyperparameter management for easy experimentation.
"""

from dataclasses import dataclass, field


@dataclass
class RLConfig:
    """Configuration for RL Navigator."""

    # === Model Architecture ===
    encoder_name: str = "dinov2_vits14"  # or dinov2_vitb14, dinov2_vitl14
    policy_hidden_dim: int = 256
    predictor_hidden_dim: int = 512

    # === Training Phases ===
    phase1_episodes: int = 10000  # Frozen encoder
    phase2_episodes: int = 5000   # Fine-tuned encoder
    max_steps_per_episode: int = 500

    # === RL Hyperparameters ===
    gamma: float = 0.99           # Discount factor
    gae_lambda: float = 0.95      # GAE lambda
    clip_epsilon: float = 0.2     # PPO clipping

    # === Learning Rates ===
    policy_lr: float = 3e-4
    predictor_lr: float = 1e-3
    encoder_lr: float = 1e-5      # Very small for Phase 2!

    # === Training Configuration ===
    rollout_steps: int = 2048     # Steps per rollout
    ppo_epochs: int = 4           # PPO update epochs
    ppo_batch_size: int = 64
    max_grad_norm: float = 0.5

    # === Intrinsic Motivation ===
    use_rnd: bool = False         # Use RND instead of forward dynamics
    reward_horizon: int = 10      # Look-ahead horizon
    reward_lambda: float = 0.1    # Exponential decay rate
    coverage_bonus_weight: float = 0.1

    # === Logging & Checkpointing ===
    log_interval: int = 10        # Episodes between logs
    save_interval: int = 1000     # Episodes between checkpoints
    save_dir: str = "checkpoints/rl_navigator"

    # === Device ===
    device: str = "cuda"  # or "cpu"


@dataclass
class QuickTestConfig(RLConfig):
    """Fast configuration for testing/debugging."""

    phase1_episodes: int = 100
    phase2_episodes: int = 50
    max_steps_per_episode: int = 100
    rollout_steps: int = 512
    log_interval: int = 5
    save_interval: int = 50


@dataclass
class RNDConfig(RLConfig):
    """Configuration using RND instead of forward dynamics."""

    use_rnd: bool = True
    predictor_hidden_dim: int = 512


@dataclass
class LongTrainingConfig(RLConfig):
    """Extended training configuration."""

    phase1_episodes: int = 20000
    phase2_episodes: int = 10000
    encoder_name: str = "dinov2_vitb14"  # Larger model


# Default configurations
CONFIGS = {
    "default": RLConfig(),
    "quick_test": QuickTestConfig(),
    "rnd": RNDConfig(),
    "long": LongTrainingConfig(),
}


def get_config(name="default"):
    """
    Get configuration by name.

    Args:
        name: Configuration name ("default", "quick_test", "rnd", "long")

    Returns:
        config: RLConfig instance
    """
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")

    return CONFIGS[name]
