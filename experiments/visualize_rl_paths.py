"""
Visualize RL-Learned Navigation Paths

Loads trained RL navigator and visualizes the paths it learns on test images.

Usage:
    python experiments/visualize_rl_paths.py --checkpoint checkpoints/rl_navigator/final.pt --image test.jpg
"""

import argparse
import sys
from pathlib import Path

sys.path.append("..")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from techniques.rl_navigation import (
    ImageNavigationEnv,
    NavigationPolicy,
    SemanticEncoder,
)
from techniques.rl_navigation.forward_dynamics import (
    ForwardDynamicsModel,
    RNDIntrinsicMotivation,
)


def load_image(image_path, max_size=512):
    """Load and preprocess image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize if needed
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        image = cv2.resize(image, (new_w, new_h))

    return image


def visualize_path(image, path, title="RL-Learned Path", color=(255, 0, 0)):
    """
    Visualize navigation path on image.

    Args:
        image: RGB image (H, W, 3)
        path: List of (row, col) positions
        title: Plot title
        color: Path color (RGB)

    Returns:
        vis_image: Image with path drawn
    """
    vis_image = image.copy()

    # Draw path
    for i in range(len(path) - 1):
        pt1 = (path[i][1], path[i][0])  # (col, row)
        pt2 = (path[i + 1][1], path[i + 1][0])
        cv2.line(vis_image, pt1, pt2, color, 2)

    # Mark start (green) and end (red)
    start_pt = (path[0][1], path[0][0])
    end_pt = (path[-1][1], path[-1][0])
    cv2.circle(vis_image, start_pt, 5, (0, 255, 0), -1)
    cv2.circle(vis_image, end_pt, 5, (0, 0, 255), -1)

    return vis_image


def run_episode(env, policy, deterministic=True, max_steps=500):
    """
    Run single episode with policy.

    Args:
        env: ImageNavigationEnv
        policy: NavigationPolicy
        deterministic: If True, use argmax; else sample
        max_steps: Maximum steps

    Returns:
        path: List of (row, col) positions
        total_reward: Cumulative reward
        stats: Episode statistics
    """
    policy.eval()

    state = env.reset()
    path = [state["position"]]
    total_reward = 0
    step_count = 0

    for _ in range(max_steps):
        # Get features
        features = state["features"].to(env.device)

        # Select action
        with torch.no_grad():
            action, _, _ = policy.act(features, deterministic=deterministic)

        # Execute
        state, reward, done, info = env.step(action.item())

        path.append(state["position"])
        total_reward += reward
        step_count += 1

        if done:
            break

    stats = env.get_statistics()
    stats["total_reward"] = total_reward

    return path, total_reward, stats


def compare_multiple_paths(image, paths, labels, save_path=None):
    """
    Compare multiple navigation paths.

    Args:
        image: RGB image
        paths: List of path lists
        labels: List of labels for each path
        save_path: Optional path to save visualization
    """
    n_paths = len(paths)
    fig, axes = plt.subplots(1, n_paths, figsize=(6 * n_paths, 6))

    if n_paths == 1:
        axes = [axes]

    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
    ]

    for idx, (path, label) in enumerate(zip(paths, labels)):
        color = colors[idx % len(colors)]
        vis = visualize_path(image, path, title=label, color=color)

        axes[idx].imshow(vis)
        axes[idx].set_title(f"{label}\n({len(path)} steps)")
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize RL-Learned Paths")

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained checkpoint"
    )
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--n_episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic policy"
    )
    parser.add_argument("--save_dir", type=str, default="experiments/outputs/rl_paths")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("RL Path Visualization")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}\n")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Load image
    if args.image:
        image = load_image(args.image)
        print(f"Loaded image: {args.image}")
    else:
        # Create test image
        from experiments.train_rl_navigator import create_test_image

        image = create_test_image()
        print("Using synthetic test image")

    print(f"Image shape: {image.shape}\n")

    # Initialize components
    print("Initializing models...")

    # Encoder (need to match checkpoint)
    encoder = SemanticEncoder(
        model_name="dinov2_vits14",  # Adjust based on your checkpoint
        freeze=True,
        device=args.device,
    )
    encoder.load_state_dict(checkpoint["encoder_state_dict"])

    # Policy
    policy = NavigationPolicy(
        feature_dim=encoder.feature_dim, action_dim=8, hidden_dim=256
    ).to(args.device)
    policy.load_state_dict(checkpoint["policy_state_dict"])

    # Predictor (need to determine if RND or Forward Dynamics)
    # Try to infer from checkpoint keys
    predictor_keys = checkpoint["predictor_state_dict"].keys()
    if any("target_net" in key for key in predictor_keys):
        predictor = RNDIntrinsicMotivation(feature_dim=encoder.feature_dim).to(
            args.device
        )
        print("Using RND")
    else:
        predictor = ForwardDynamicsModel(
            feature_dim=encoder.feature_dim, action_dim=8
        ).to(args.device)
        print("Using Forward Dynamics")

    predictor.load_state_dict(checkpoint["predictor_state_dict"])

    # Environment
    env = ImageNavigationEnv(
        image=image, encoder=encoder, predictor=predictor, device=args.device
    )

    print("Models loaded!\n")

    # Run multiple episodes
    print(f"Running {args.n_episodes} episodes...\n")

    all_paths = []
    all_stats = []

    for ep in range(args.n_episodes):
        path, total_reward, stats = run_episode(
            env, policy, deterministic=args.deterministic
        )

        all_paths.append(path)
        all_stats.append(stats)

        print(f"Episode {ep + 1}:")
        print(f"  Steps: {len(path)}")
        print(f"  Reward: {total_reward:.2f}")
        print(f"  Coverage: {stats['coverage']:.2%}")
        print()

        # Save individual visualization
        vis = visualize_path(image, path, title=f"RL Path (Episode {ep+1})")
        save_path = Path(args.save_dir) / f"rl_path_ep{ep+1}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    # Compare first few paths
    labels = [f"Episode {i+1}" for i in range(min(4, args.n_episodes))]
    compare_multiple_paths(
        image,
        all_paths[: min(4, args.n_episodes)],
        labels,
        save_path=Path(args.save_dir) / "comparison.png",
    )

    # Print average statistics
    print("\n" + "=" * 60)
    print("Average Statistics:")
    print("=" * 60)
    avg_steps = np.mean([len(p) for p in all_paths])
    avg_coverage = np.mean([s["coverage"] for s in all_stats])
    avg_reward = np.mean([s["total_reward"] for s in all_stats])

    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Avg Coverage: {avg_coverage:.2%}")
    print(f"Avg Reward: {avg_reward:.2f}")

    print(f"\nVisualizations saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
