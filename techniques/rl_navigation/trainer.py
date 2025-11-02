"""
Two-Phase RL Trainer for Image Navigation

Orchestrates co-training of:
- Semantic Encoder (E): DINOv2 → frozen → fine-tuned
- Navigation Policy (π): PPO-based learning
- Forward Dynamics (P): Intrinsic motivation

Phase 1: Frozen encoder, stable semantic space
Phase 2: Fine-tune encoder top layers for task-specific features
"""

import os
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .encoder import SemanticEncoder
from .environment import ImageNavigationEnv
from .forward_dynamics import ForwardDynamicsModel, RNDIntrinsicMotivation
from .policy import NavigationPolicy, PPOTrainer


class RLTrainer:
    """
    Two-phase trainer for RL-based image navigation.

    Manages:
    - Phase 1: Frozen encoder (10k episodes)
    - Phase 2: Fine-tuned encoder (5k episodes)
    - Rollout collection
    - Policy updates (PPO)
    - Forward model updates
    - Logging and checkpointing
    """

    def __init__(
        self,
        image,
        encoder_name="dinov2_vits14",
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Phase configuration
        phase1_episodes=10000,
        phase2_episodes=5000,
        # Network hyperparameters
        policy_hidden_dim=256,
        predictor_hidden_dim=512,
        # RL hyperparameters
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        # Learning rates
        policy_lr=3e-4,
        predictor_lr=1e-3,
        encoder_lr=1e-5,
        # Training config
        max_steps_per_episode=500,
        rollout_steps=2048,
        ppo_epochs=4,
        ppo_batch_size=64,
        # Intrinsic motivation
        use_rnd=False,
        reward_horizon=10,
        reward_lambda=0.1,
        coverage_bonus_weight=0.1,
        # Logging
        log_interval=10,
        save_interval=1000,
        save_dir="checkpoints/rl_navigator",
    ):
        """
        Args:
            image: RGB image to navigate (numpy array or tensor)
            encoder_name: DINOv2 model variant
            device: Training device
            phase1_episodes: Number of episodes with frozen encoder
            phase2_episodes: Number of episodes with fine-tuned encoder
            ... (see parameters above)
        """
        self.device = device
        self.phase1_episodes = phase1_episodes
        self.phase2_episodes = phase2_episodes
        self.rollout_steps = rollout_steps
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.save_dir = save_dir

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Initialize components
        print("=== Initializing RL Navigator ===")

        # 1. Semantic Encoder (frozen initially)
        self.encoder = SemanticEncoder(
            model_name=encoder_name,
            freeze=True,
            device=device,
        )
        feature_dim = self.encoder.feature_dim

        # 2. Navigation Policy
        self.policy = NavigationPolicy(
            feature_dim=feature_dim,
            action_dim=8,
            hidden_dim=policy_hidden_dim,
        ).to(device)

        # 3. Forward Dynamics or RND
        if use_rnd:
            self.predictor = RNDIntrinsicMotivation(
                feature_dim=feature_dim,
                hidden_dim=predictor_hidden_dim,
            ).to(device)
            print("Using RND for intrinsic motivation")
        else:
            self.predictor = ForwardDynamicsModel(
                feature_dim=feature_dim,
                action_dim=8,
                hidden_dim=predictor_hidden_dim,
            ).to(device)
            print("Using Forward Dynamics for intrinsic motivation")

        # 4. Environment
        self.env = ImageNavigationEnv(
            image=image,
            encoder=self.encoder,
            predictor=self.predictor,
            max_steps=max_steps_per_episode,
            reward_horizon=reward_horizon,
            reward_lambda=reward_lambda,
            coverage_bonus_weight=coverage_bonus_weight,
            device=device,
        )

        # 5. Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=predictor_lr)
        self.encoder_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.encoder.parameters()),
            lr=encoder_lr,
        )

        # 6. PPO Trainer
        self.ppo_trainer = PPOTrainer(
            policy=self.policy,
            optimizer=self.policy_optimizer,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_epsilon=clip_epsilon,
        )

        # Training state
        self.episode_count = 0
        self.phase = 1
        self.rollout_buffer = self._init_rollout_buffer()

        # Metrics tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.coverage_stats = deque(maxlen=100)
        self.prediction_errors = deque(maxlen=100)

        print(f"Feature dim: {feature_dim}")
        print(f"Device: {device}")
        print("=== Initialization Complete ===\n")

    def _init_rollout_buffer(self):
        """Initialize rollout buffer for collecting experiences."""
        return {
            "features": [],
            "actions": [],
            "old_log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
        }

    def collect_rollout(self):
        """
        Collect rollout of experiences.

        Returns:
            rollout_buffer: Dictionary with experiences
        """
        self.policy.eval()
        self.rollout_buffer = self._init_rollout_buffer()

        state = self.env.reset()
        episode_reward = 0
        step_count = 0

        for _ in range(self.rollout_steps):
            # Get features
            features = state["features"].to(self.device)

            # Select action (detached features for policy!)
            with torch.no_grad():
                action, log_prob, value = self.policy.act(features.detach())

            # Execute action
            next_state, reward, done, info = self.env.step(action.item())

            # Store transition
            self.rollout_buffer["features"].append(features)
            self.rollout_buffer["actions"].append(action)
            self.rollout_buffer["old_log_probs"].append(log_prob)
            self.rollout_buffer["rewards"].append(torch.tensor(reward, device=self.device))
            self.rollout_buffer["values"].append(value)
            self.rollout_buffer["dones"].append(torch.tensor(done, dtype=torch.float32, device=self.device))

            episode_reward += reward
            step_count += 1

            if done:
                # Episode ended
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(step_count)
                self.coverage_stats.append(self.env.get_statistics()["coverage"])

                # Reset for next episode
                state = self.env.reset()
                episode_reward = 0
                step_count = 0
            else:
                state = next_state

        # Add final value estimate
        with torch.no_grad():
            final_features = state["features"].to(self.device)
            _, _, final_value = self.policy.act(final_features.detach())
        self.rollout_buffer["values"].append(final_value)

        # Convert lists to tensors
        for key in self.rollout_buffer:
            if key != "values":
                self.rollout_buffer[key] = torch.stack(self.rollout_buffer[key])
            else:
                self.rollout_buffer[key] = torch.stack(self.rollout_buffer[key])

        return self.rollout_buffer

    def update_predictor(self):
        """Update forward dynamics model or RND."""
        self.predictor.train()

        # Sample batch from rollout buffer
        features_t = self.rollout_buffer["features"]
        actions = self.rollout_buffer["actions"]

        # Get next features
        features_t1 = torch.cat([features_t[1:], features_t[-1:]])

        # Update predictor
        if isinstance(self.predictor, RNDIntrinsicMotivation):
            loss = self.predictor.update(features_t1, self.predictor_optimizer)
        else:
            loss = self.predictor.update(features_t, actions, features_t1, self.predictor_optimizer)

        self.prediction_errors.append(loss)

        return loss

    def train_step(self):
        """Single training step: collect rollout, update policy and predictor."""
        # Collect experiences
        rollout = self.collect_rollout()

        # Update policy with PPO
        self.policy.train()
        ppo_metrics = self.ppo_trainer.update(
            rollout,
            n_epochs=self.ppo_epochs,
            batch_size=self.ppo_batch_size,
        )

        # Update predictor
        predictor_loss = self.update_predictor()

        # Update encoder if in Phase 2
        if self.phase == 2:
            # Encoder gradients flow through predictor updates
            # (already handled in predictor.update)
            pass

        return {
            **ppo_metrics,
            "predictor_loss": predictor_loss,
        }

    def train(self):
        """
        Full two-phase training loop.

        Phase 1: Frozen encoder
        Phase 2: Fine-tune encoder
        """
        print("\n" + "=" * 60)
        print("PHASE 1: Training with Frozen Encoder")
        print("=" * 60)
        self.phase = 1
        self.encoder.freeze()

        for episode in range(self.phase1_episodes):
            metrics = self.train_step()
            self.episode_count += 1

            # Logging
            if (episode + 1) % self.log_interval == 0:
                self._log_progress(episode + 1, self.phase1_episodes, metrics)

            # Checkpointing
            if (episode + 1) % self.save_interval == 0:
                self.save_checkpoint(f"phase1_ep{episode+1}.pt")

        print("\n" + "=" * 60)
        print("PHASE 2: Fine-Tuning Encoder")
        print("=" * 60)
        self.phase = 2
        self.encoder.unfreeze_top_layers(n_layers=2)

        for episode in range(self.phase2_episodes):
            metrics = self.train_step()
            self.episode_count += 1

            # Logging
            if (episode + 1) % self.log_interval == 0:
                self._log_progress(episode + 1, self.phase2_episodes, metrics)

            # Checkpointing
            if (episode + 1) % self.save_interval == 0:
                self.save_checkpoint(f"phase2_ep{episode+1}.pt")

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        self.save_checkpoint("final.pt")

    def _log_progress(self, episode, total_episodes, metrics):
        """Log training progress."""
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        avg_coverage = np.mean(self.coverage_stats) if self.coverage_stats else 0
        avg_pred_error = np.mean(self.prediction_errors) if self.prediction_errors else 0

        print(f"\nPhase {self.phase} - Episode {episode}/{total_episodes}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Length: {avg_length:.1f}")
        print(f"  Coverage: {avg_coverage:.2%}")
        print(f"  Pred Error: {avg_pred_error:.4f}")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"  Value Loss: {metrics['value_loss']:.4f}")
        print(f"  Entropy: {metrics['entropy']:.4f}")

    def save_checkpoint(self, filename):
        """Save training checkpoint."""
        checkpoint = {
            "episode": self.episode_count,
            "phase": self.phase,
            "policy_state_dict": self.policy.state_dict(),
            "predictor_state_dict": self.predictor.state_dict(),
            "encoder_state_dict": self.encoder.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "predictor_optimizer_state_dict": self.predictor_optimizer.state_dict(),
            "metrics": {
                "episode_rewards": list(self.episode_rewards),
                "episode_lengths": list(self.episode_lengths),
                "coverage_stats": list(self.coverage_stats),
                "prediction_errors": list(self.prediction_errors),
            },
        }

        save_path = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")

    def load_checkpoint(self, filename):
        """Load training checkpoint."""
        load_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(load_path, map_location=self.device)

        self.episode_count = checkpoint["episode"]
        self.phase = checkpoint["phase"]
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.predictor.load_state_dict(checkpoint["predictor_state_dict"])
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.predictor_optimizer.load_state_dict(checkpoint["predictor_optimizer_state_dict"])

        print(f"Checkpoint loaded: {load_path}")
        print(f"Resuming from Episode {self.episode_count}, Phase {self.phase}")
