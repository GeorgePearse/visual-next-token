"""
Train RL Navigator for Image Exploration

Learns to navigate images by maximizing prediction error of future semantic
features, forcing exploration of information-dense regions.

Usage:
    # Quick test
    python experiments/train_rl_navigator.py --config quick_test

    # Full training
    python experiments/train_rl_navigator.py --config default

    # With custom image
    python experiments/train_rl_navigator.py --image path/to/image.jpg

    # Resume from checkpoint
    python experiments/train_rl_navigator.py --resume checkpoints/rl_navigator/phase1_ep1000.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.append("..")

import cv2
import numpy as np
import torch

from techniques.rl_navigation import RLTrainer
from techniques.rl_navigation.config import get_config


def load_image(image_path):
    """
    Load and preprocess image.

    Args:
        image_path: Path to image file

    Returns:
        image: RGB image as numpy array (H, W, 3)
    """
    # Read image
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize if too large (for memory)
    max_size = 512
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))
        print(f"Resized image to {new_h}x{new_w}")

    return image


def create_test_image(size=(256, 256)):
    """Create synthetic test image with various features."""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    # Gradient background
    for i in range(size[0]):
        img[i, :, :] = [i * 255 // size[0], 128, 255 - i * 255 // size[0]]

    # Shapes
    cv2.circle(img, (size[1] // 4, size[0] // 4), 30, (255, 0, 0), -1)
    cv2.rectangle(
        img,
        (size[1] // 2, size[0] // 2),
        (size[1] // 2 + 60, size[0] // 2 + 60),
        (0, 255, 0),
        -1,
    )
    cv2.circle(img, (3 * size[1] // 4, 3 * size[0] // 4), 40, (0, 0, 255), -1)

    return img


def main():
    parser = argparse.ArgumentParser(description="Train RL Navigator")

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=["default", "quick_test", "rnd", "long"],
        help="Configuration preset",
    )

    # Image
    parser.add_argument("--image", type=str, help="Path to image file")

    # Checkpointing
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--save_dir", type=str, help="Override save directory")

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    # Get configuration
    config = get_config(args.config)
    config.device = args.device

    if args.save_dir:
        config.save_dir = args.save_dir

    print("=" * 60)
    print("RL Navigator Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {config.device}")
    print(f"Encoder: {config.encoder_name}")
    print(f"Phase 1 Episodes: {config.phase1_episodes}")
    print(f"Phase 2 Episodes: {config.phase2_episodes}")
    print("=" * 60 + "\n")

    # Load or create image
    if args.image:
        print(f"Loading image: {args.image}")
        image = load_image(args.image)
    else:
        print("Creating synthetic test image")
        image = create_test_image()

    print(f"Image shape: {image.shape}\n")

    # Create trainer
    trainer = RLTrainer(
        image=image,
        encoder_name=config.encoder_name,
        device=config.device,
        phase1_episodes=config.phase1_episodes,
        phase2_episodes=config.phase2_episodes,
        policy_hidden_dim=config.policy_hidden_dim,
        predictor_hidden_dim=config.predictor_hidden_dim,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_epsilon=config.clip_epsilon,
        policy_lr=config.policy_lr,
        predictor_lr=config.predictor_lr,
        encoder_lr=config.encoder_lr,
        max_steps_per_episode=config.max_steps_per_episode,
        rollout_steps=config.rollout_steps,
        ppo_epochs=config.ppo_epochs,
        ppo_batch_size=config.ppo_batch_size,
        use_rnd=config.use_rnd,
        reward_horizon=config.reward_horizon,
        reward_lambda=config.reward_lambda,
        coverage_bonus_weight=config.coverage_bonus_weight,
        log_interval=config.log_interval,
        save_interval=config.save_interval,
        save_dir=config.save_dir,
    )

    # Resume if checkpoint provided
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted.pt")
        print("Checkpoint saved!")

    print("\nTraining complete!")
    print(f"Final checkpoint saved to: {config.save_dir}/final.pt")


if __name__ == "__main__":
    main()
