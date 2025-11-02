"""
Demo: Walk Tokenization for Next-Token Prediction

Demonstrates converting image walks into discrete tokens that predict CHANGE:
- Direction: Where will the walk move next?
- Delta: How much will the pixel value change?

This is the image equivalent of "next token prediction" in language models.

Usage:
    python experiments/demo_walk_tokenization.py
"""

import sys
from pathlib import Path

sys.path.append("..")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.image_walker import (
    BrightnessGradientWalk,
    EdgeFollowingWalk,
    ImageWalker,
    SaliencyWalk,
    StochasticGradientWalk,
)
from utils.walk_tokenizer import Direction, WalkTokenizer, visualize_token_sequence


def create_test_image(size=(256, 256)):
    """Create a test image with various features."""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)

    # Add a gradient background
    for i in range(size[0]):
        img[i, :, :] = [i * 255 // size[0], 128, 255 - i * 255 // size[0]]

    # Add some shapes
    cv2.circle(img, (size[1] // 4, size[0] // 4), 30, (255, 0, 0), -1)
    cv2.rectangle(
        img, (size[1] // 2, size[0] // 2), (size[1] // 2 + 60, size[0] // 2 + 60), (0, 255, 0), -1
    )
    cv2.circle(img, (3 * size[1] // 4, 3 * size[0] // 4), 40, (0, 0, 255), -1)

    return img


def demo_basic_tokenization(image: np.ndarray):
    """Demonstrate basic walk tokenization."""

    print("\n" + "=" * 60)
    print("DEMO 1: Basic Walk Tokenization")
    print("=" * 60)

    walker = ImageWalker(image)
    tokenizer = WalkTokenizer(n_value_bins=32, value_range=(-255, 255))

    # Execute a walk
    strategy = BrightnessGradientWalk(maximize=True)
    path = walker.walk(strategy, max_steps=500)

    print(f"\nExecuted walk with {len(path)} steps")

    # Tokenize
    tokens = tokenizer.tokenize_walk(path)
    print(f"Generated {len(tokens)} tokens (transitions)")

    # Show first few tokens
    print("\nFirst 5 tokens:")
    for i, token in enumerate(tokens[:5]):
        print(f"  Step {i + 1}: {token.direction.name:12s} -> Δ={token.value_delta}")

    # Get statistics
    stats = tokenizer.get_statistics(path)
    print("\nWalk Statistics:")
    print(f"  Total transitions: {stats['total_steps']}")
    print(f"  Mean delta: {stats['delta_mean']}")
    print(f"  Std delta: {stats['delta_std']}")
    print("  Direction distribution:")
    for direction, count in sorted(stats["direction_distribution"].items(), key=lambda x: -x[1])[
        :5
    ]:
        print(f"    {direction.name:12s}: {count:3d} ({100 * count / stats['total_steps']:.1f}%)")

    # Visualize
    viz = visualize_token_sequence(tokens, image)
    Path("experiments/outputs").mkdir(parents=True, exist_ok=True)
    cv2.imwrite("experiments/outputs/token_visualization.png", cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))
    print("\nSaved: experiments/outputs/token_visualization.png")


def demo_prediction_dataset(image: np.ndarray):
    """Demonstrate creating dataset for next-token prediction."""

    print("\n" + "=" * 60)
    print("DEMO 2: Next-Token Prediction Dataset")
    print("=" * 60)

    walker = ImageWalker(image)
    tokenizer = WalkTokenizer(n_value_bins=32)

    # Execute multiple walks with different strategies
    strategies = {
        "Gradient Max": BrightnessGradientWalk(maximize=True),
        "Gradient Min": BrightnessGradientWalk(maximize=False),
        "Stochastic": StochasticGradientWalk(temperature=1.0),
        "Edge Follow": EdgeFollowingWalk(),
    }

    all_samples = []

    for name, strategy in strategies.items():
        print(f"\nProcessing {name} walk...")
        path = walker.walk(strategy, max_steps=300)

        # Create prediction dataset with context window of 8
        samples = tokenizer.create_prediction_dataset(path, context_length=8)

        print(f"  Generated {len(samples)} training samples")
        all_samples.extend(samples)

    print(f"\nTotal samples across all walks: {len(all_samples)}")

    # Show example sample
    if all_samples:
        sample = all_samples[0]
        print("\nExample training sample:")
        print(
            f"  Context directions: {[Direction(d).name for d in sample['context_directions'][:3]]}..."
        )
        print(f"  Target direction: {Direction(sample['target_direction']).name}")
        print(f"  Target delta: {sample['target_delta']}")


def demo_direction_analysis(image: np.ndarray):
    """Analyze direction preferences of different walk strategies."""

    print("\n" + "=" * 60)
    print("DEMO 3: Direction Analysis Across Strategies")
    print("=" * 60)

    walker = ImageWalker(image)
    tokenizer = WalkTokenizer()

    strategies = {
        "Gradient Max": BrightnessGradientWalk(maximize=True),
        "Gradient Min": BrightnessGradientWalk(maximize=False),
        "Saliency": SaliencyWalk(),
        "Stochastic (T=0.5)": StochasticGradientWalk(temperature=0.5),
        "Stochastic (T=2.0)": StochasticGradientWalk(temperature=2.0),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, strategy) in enumerate(strategies.items()):
        path = walker.walk(strategy, max_steps=400)
        stats = tokenizer.get_statistics(path)

        # Plot direction distribution
        directions = list(range(9))  # 0-7 plus TERMINATE
        counts = [stats["direction_distribution"].get(Direction(d), 0) for d in directions]
        labels = [Direction(d).name for d in directions]

        axes[idx].bar(range(len(counts)), counts, tick_label=labels)
        axes[idx].set_title(f"{name}\n({stats['total_steps']} steps)")
        axes[idx].set_ylabel("Count")
        axes[idx].tick_params(axis="x", rotation=45, labelsize=8)

    # Hide last subplot if odd number of strategies
    if len(strategies) < 6:
        axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig("experiments/outputs/direction_analysis.png", dpi=150, bbox_inches="tight")
    print("\nSaved: experiments/outputs/direction_analysis.png")
    plt.close()


def demo_delta_distributions(image: np.ndarray):
    """Analyze value delta distributions."""

    print("\n" + "=" * 60)
    print("DEMO 4: Value Delta Distributions")
    print("=" * 60)

    walker = ImageWalker(image)
    tokenizer = WalkTokenizer()

    strategies = {
        "Max Gradient": BrightnessGradientWalk(maximize=True),
        "Min Gradient": BrightnessGradientWalk(maximize=False),
        "Stochastic": StochasticGradientWalk(temperature=1.0),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (name, strategy) in enumerate(strategies.items()):
        path = walker.walk(strategy, max_steps=500)
        tokens = tokenizer.tokenize_walk(path)

        # Extract deltas (for grayscale, use mean across RGB)
        deltas = []
        for token in tokens:
            if isinstance(token.value_delta, np.ndarray):
                delta = token.value_delta.mean()
            else:
                delta = token.value_delta
            deltas.append(delta)

        axes[idx].hist(deltas, bins=50, edgecolor="black", alpha=0.7)
        axes[idx].set_title(f"{name}\nμ={np.mean(deltas):.1f}, σ={np.std(deltas):.1f}")
        axes[idx].set_xlabel("Pixel Value Change (Δ)")
        axes[idx].set_ylabel("Frequency")
        axes[idx].axvline(0, color="red", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig("experiments/outputs/delta_distributions.png", dpi=150, bbox_inches="tight")
    print("\nSaved: experiments/outputs/delta_distributions.png")
    plt.close()


def main():
    """Run all tokenization demos."""

    # Create output directory
    Path("experiments/outputs").mkdir(parents=True, exist_ok=True)

    # Load or create test image
    try:
        image = cv2.imread("experiments/test_image.jpg")
        if image is None:
            raise FileNotFoundError
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Loaded test_image.jpg")
    except FileNotFoundError:
        image = create_test_image()
        cv2.imwrite("experiments/test_image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("Created synthetic test image")

    print(f"Image shape: {image.shape}")

    # Run demos
    demo_basic_tokenization(image)
    demo_prediction_dataset(image)
    demo_direction_analysis(image)
    demo_delta_distributions(image)

    print("\n" + "=" * 60)
    print("All tokenization demos complete!")
    print("Check experiments/outputs/ for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
