"""
Demo: Image Walking Strategies

Demonstrates different walk strategies through images and visualizes the paths.

Usage:
    python experiments/demo_image_walks.py --image path/to/image.jpg
"""

import sys

sys.path.append("..")

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils.image_walker import (
    BrightnessGradientWalk,
    CenterOutwardWalk,
    ColorChannelGradientWalk,
    ContourWigglingWalk,
    EdgeFollowingWalk,
    ImageWalker,
    NoBacktrackingMinChangeWalk,
    RandomWalk,
    SaliencyWalk,
    SpiralWalk,
    StochasticGradientWalk,
)
from utils.superpixel_walker import SuperpixelWalker


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


def demo_pixel_walks(image: np.ndarray, max_steps: int = 1000):
    """Demonstrate different pixel-level walk strategies."""

    strategies = [
        ("Brightness Max Gradient", BrightnessGradientWalk(maximize=True)),
        ("Brightness Min Gradient", BrightnessGradientWalk(maximize=False)),
        ("Stochastic (T=1.0)", StochasticGradientWalk(temperature=1.0)),
        ("No-Backtrack Min Change", NoBacktrackingMinChangeWalk()),
        ("Red Channel Max", ColorChannelGradientWalk(channel=0, maximize=True)),
        ("Saliency Walk", SaliencyWalk()),
        ("Center Outward", CenterOutwardWalk(reverse=False)),
        ("Spiral", SpiralWalk()),
        ("Edge Following", EdgeFollowingWalk()),
        ("Contour Wiggling", ContourWigglingWalk(wiggle_amplitude=3)),
        ("Random (Baseline)", RandomWalk()),
    ]

    len(strategies)
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    axes = axes.flatten()

    walker = ImageWalker(image)

    for idx, (name, strategy) in enumerate(strategies):
        print(f"Running: {name}...")

        # Execute walk
        path = walker.walk(strategy, max_steps=max_steps)

        # Visualize
        viz = walker.visualize(path, line_thickness=1, color=(255, 255, 0))

        # Display
        axes[idx].imshow(viz)
        axes[idx].set_title(f"{name}\n({len(path)} steps)")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig("experiments/outputs/pixel_walks_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: experiments/outputs/pixel_walks_comparison.png")
    plt.close()


def demo_superpixel_walks(image: np.ndarray):
    """Demonstrate superpixel-level walks."""

    # Different segmentation methods
    methods = ["slic", "felzenszwalb", "quickshift", "watershed"]

    fig, axes = plt.subplots(len(methods), 4, figsize=(16, 4 * len(methods)))

    for method_idx, method in enumerate(methods):
        print(f"\nSegmentation method: {method.upper()}")

        try:
            sp_walker = SuperpixelWalker(image, method=method, n_segments=50)

            # Different ordering strategies
            orderings = [
                ("By Size (Largest First)", sp_walker.walk_by_size(largest_first=True)),
                ("By Brightness", sp_walker.walk_by_brightness(brightest_first=True)),
                ("By Position (Center Out)", sp_walker.walk_by_position("center")),
                ("By Gradient (Max)", sp_walker.walk_by_gradient(maximize=True)),
            ]

            for order_idx, (order_name, order) in enumerate(orderings):
                viz = sp_walker.visualize_superpixels(order, show_order=True)

                axes[method_idx, order_idx].imshow(viz)
                axes[method_idx, order_idx].set_title(f"{method}: {order_name}")
                axes[method_idx, order_idx].axis("off")

        except Exception as e:
            print(f"Error with {method}: {e}")
            for order_idx in range(4):
                axes[method_idx, order_idx].text(
                    0.5, 0.5, f"Error:\n{str(e)}", ha="center", va="center"
                )
                axes[method_idx, order_idx].axis("off")

    plt.tight_layout()
    plt.savefig("experiments/outputs/superpixel_walks_comparison.png", dpi=150, bbox_inches="tight")
    print("\nSaved: experiments/outputs/superpixel_walks_comparison.png")
    plt.close()


def demo_walk_statistics(image: np.ndarray, max_steps: int = 500):
    """Analyze statistics of different walk strategies."""

    strategies = {
        "Max Gradient": BrightnessGradientWalk(maximize=True),
        "Min Gradient": BrightnessGradientWalk(maximize=False),
        "Min Change (No Backtrack)": NoBacktrackingMinChangeWalk(),
        "Saliency": SaliencyWalk(),
        "Random": RandomWalk(),
    }

    walker = ImageWalker(image)

    stats = {}
    for name, strategy in strategies.items():
        path = walker.walk(strategy, max_steps=max_steps)

        # Compute statistics
        gradients = [step.gradient_magnitude for step in path]
        colors = [
            step.value.mean() if isinstance(step.value, np.ndarray) else step.value for step in path
        ]

        stats[name] = {
            "path_length": len(path),
            "avg_gradient": np.mean(gradients) if gradients else 0,
            "avg_color": np.mean(colors),
            "std_color": np.std(colors),
        }

    # Print statistics
    print("\n=== Walk Statistics ===")
    print(f"{'Strategy':<30} {'Steps':<10} {'Avg Grad':<15} {'Avg Color':<15} {'Color StdDev':<15}")
    print("-" * 85)
    for name, stat in stats.items():
        print(
            f"{name:<30} {stat['path_length']:<10} "
            f"{stat['avg_gradient']:<15.2f} "
            f"{stat['avg_color']:<15.2f} "
            f"{stat['std_color']:<15.2f}"
        )


def main():
    """Run all demos."""

    # Create output directory
    Path("experiments/outputs").mkdir(parents=True, exist_ok=True)

    # Load or create test image
    try:
        # Try to load an example image
        image = cv2.imread("experiments/test_image.jpg")
        if image is None:
            raise FileNotFoundError
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Loaded test_image.jpg")
    except (FileNotFoundError, cv2.error):
        # Create synthetic test image
        image = create_test_image()
        cv2.imwrite("experiments/test_image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("Created synthetic test image")

    print(f"Image shape: {image.shape}")

    # Run demos
    print("\n" + "=" * 60)
    print("DEMO 1: Pixel-Level Walks")
    print("=" * 60)
    demo_pixel_walks(image, max_steps=1000)

    print("\n" + "=" * 60)
    print("DEMO 2: Superpixel-Level Walks")
    print("=" * 60)
    demo_superpixel_walks(image)

    print("\n" + "=" * 60)
    print("DEMO 3: Walk Statistics")
    print("=" * 60)
    demo_walk_statistics(image, max_steps=500)

    print("\n" + "=" * 60)
    print("All demos complete! Check experiments/outputs/ for visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()
