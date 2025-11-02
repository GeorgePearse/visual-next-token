"""
Image Walker Library - Visualize different traversal strategies through images.

This library implements various walk strategies to traverse images in content-aware ways:
- Gradient-based walks (follow brightness, color, edges)
- Stochastic walks (SGD-style with randomization)
- Saliency-based walks
- Superpixel-based walks

Usage:
    from utils.image_walker import ImageWalker, BrightnessGradientWalk

    walker = ImageWalker(image)
    path = walker.walk(BrightnessGradientWalk())
    walker.visualize(path)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class WalkStep:
    """Single step in an image walk."""

    position: tuple[int, int]  # (row, col)
    value: np.ndarray  # Pixel value at this position
    gradient_magnitude: float = 0.0
    direction: tuple[int, int] | None = None  # Direction moved from previous


class WalkStrategy(ABC):
    """Abstract base class for walk strategies."""

    @abstractmethod
    def compute_score(
        self,
        image: np.ndarray,
        position: tuple[int, int],
        visited: set,
        walk_history: list[WalkStep],
    ) -> float:
        """
        Compute score for a position. Higher score = more likely to visit next.

        Args:
            image: Input image (H, W, C) or (H, W)
            position: Current position (row, col)
            visited: Set of already visited positions
            walk_history: Previous steps taken

        Returns:
            Score for this position
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable name of this strategy."""
        pass

    def get_neighbors(
        self,
        position: tuple[int, int],
        image_shape: tuple[int, int],
        visited: set,
        connectivity: int = 8,
    ) -> list[tuple[int, int]]:
        """
        Get valid unvisited neighbors.

        Args:
            position: Current position
            image_shape: (height, width)
            visited: Already visited positions
            connectivity: 4 or 8 connected neighbors
        """
        row, col = position
        h, w = image_shape

        if connectivity == 4:
            deltas = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        else:  # 8-connected
            deltas = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        neighbors = []
        for dr, dc in deltas:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < h and 0 <= new_col < w and (new_row, new_col) not in visited:
                neighbors.append((new_row, new_col))

        return neighbors


class BrightnessGradientWalk(WalkStrategy):
    """Follow direction of greatest sum(R+G+B) change."""

    def __init__(self, maximize: bool = True):
        """
        Args:
            maximize: If True, follow max gradient. If False, follow min gradient.
        """
        self.maximize = maximize
        self._gradient_cache = None

    def compute_score(
        self,
        image: np.ndarray,
        position: tuple[int, int],
        visited: set,
        walk_history: list[WalkStep],
    ) -> float:
        if self._gradient_cache is None:
            # Convert to grayscale (sum of RGB)
            if len(image.shape) == 3:
                gray = np.sum(image, axis=2)
            else:
                gray = image

            # Compute gradient magnitude
            grad_x = ndimage.sobel(gray, axis=1)
            grad_y = ndimage.sobel(gray, axis=0)
            self._gradient_cache = np.sqrt(grad_x**2 + grad_y**2)

        row, col = position
        score = self._gradient_cache[row, col]
        return score if self.maximize else -score

    def get_name(self) -> str:
        direction = "Maximum" if self.maximize else "Minimum"
        return f"{direction} Brightness Gradient Walk"


class StochasticGradientWalk(WalkStrategy):
    """
    Follow gradient with randomization (like SGD).
    Samples from neighbors weighted by gradient magnitude.
    """

    def __init__(self, temperature: float = 1.0, maximize: bool = True):
        """
        Args:
            temperature: Higher = more random. Lower = more greedy.
            maximize: Follow max or min gradient
        """
        self.temperature = temperature
        self.maximize = maximize
        self._gradient_cache = None

    def compute_score(
        self,
        image: np.ndarray,
        position: tuple[int, int],
        visited: set,
        walk_history: list[WalkStep],
    ) -> float:
        if self._gradient_cache is None:
            if len(image.shape) == 3:
                gray = np.sum(image, axis=2)
            else:
                gray = image

            grad_x = ndimage.sobel(gray, axis=1)
            grad_y = ndimage.sobel(gray, axis=0)
            self._gradient_cache = np.sqrt(grad_x**2 + grad_y**2)

        row, col = position
        score = self._gradient_cache[row, col]

        # Add noise proportional to temperature
        noise = np.random.normal(0, self.temperature)
        score = score + noise

        return score if self.maximize else -score

    def get_name(self) -> str:
        direction = "Maximum" if self.maximize else "Minimum"
        return f"Stochastic Gradient Walk (T={self.temperature:.2f}, {direction})"


class ColorChannelGradientWalk(WalkStrategy):
    """Follow gradient in specific color channel."""

    def __init__(self, channel: int = 0, maximize: bool = True):
        """
        Args:
            channel: 0=Red, 1=Green, 2=Blue (for RGB images)
            maximize: Follow max or min gradient
        """
        self.channel = channel
        self.maximize = maximize
        self._gradient_cache = None

    def compute_score(
        self,
        image: np.ndarray,
        position: tuple[int, int],
        visited: set,
        walk_history: list[WalkStep],
    ) -> float:
        if self._gradient_cache is None:
            if len(image.shape) == 3:
                channel_image = image[:, :, self.channel]
            else:
                channel_image = image

            grad_x = ndimage.sobel(channel_image, axis=1)
            grad_y = ndimage.sobel(channel_image, axis=0)
            self._gradient_cache = np.sqrt(grad_x**2 + grad_y**2)

        row, col = position
        score = self._gradient_cache[row, col]
        return score if self.maximize else -score

    def get_name(self) -> str:
        channels = ["Red", "Green", "Blue"]
        channel_name = channels[self.channel] if self.channel < 3 else f"Channel{self.channel}"
        direction = "Max" if self.maximize else "Min"
        return f"{direction} {channel_name} Gradient Walk"


class SaliencyWalk(WalkStrategy):
    """Follow visual saliency (high-contrast regions)."""

    def __init__(self):
        self._saliency_cache = None

    def compute_score(
        self,
        image: np.ndarray,
        position: tuple[int, int],
        visited: set,
        walk_history: list[WalkStep],
    ) -> float:
        if self._saliency_cache is None:
            # Simple saliency: gradient magnitude + edge detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image.astype(np.uint8)

            # Combine gradients and edge detection
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(grad_x**2 + grad_y**2)

            # Normalize
            self._saliency_cache = (gradient - gradient.min()) / (
                gradient.max() - gradient.min() + 1e-8
            )

        row, col = position
        return self._saliency_cache[row, col]

    def get_name(self) -> str:
        return "Saliency-Based Walk"


class CenterOutwardWalk(WalkStrategy):
    """Walk from center outward (distance-based)."""

    def __init__(self, reverse: bool = False):
        """
        Args:
            reverse: If True, walk from edges to center
        """
        self.reverse = reverse
        self._distance_cache = None

    def compute_score(
        self,
        image: np.ndarray,
        position: tuple[int, int],
        visited: set,
        walk_history: list[WalkStep],
    ) -> float:
        if self._distance_cache is None:
            h, w = image.shape[:2]
            center_r, center_c = h // 2, w // 2

            # Distance from center
            rows, cols = np.ogrid[:h, :w]
            self._distance_cache = np.sqrt((rows - center_r) ** 2 + (cols - center_c) ** 2)

        row, col = position
        score = self._distance_cache[row, col]
        return -score if not self.reverse else score

    def get_name(self) -> str:
        direction = "Edges→Center" if self.reverse else "Center→Outward"
        return f"{direction} Walk"


class SpiralWalk(WalkStrategy):
    """Walk in spiral pattern from center."""

    def __init__(self):
        self._angle_cache = None
        self._radius_cache = None

    def compute_score(
        self,
        image: np.ndarray,
        position: tuple[int, int],
        visited: set,
        walk_history: list[WalkStep],
    ) -> float:
        if self._angle_cache is None:
            h, w = image.shape[:2]
            center_r, center_c = h // 2, w // 2

            rows, cols = np.ogrid[:h, :w]

            # Angle from center
            angles = np.arctan2(rows - center_r, cols - center_c)
            self._angle_cache = angles

            # Radius
            self._radius_cache = np.sqrt((rows - center_r) ** 2 + (cols - center_c) ** 2)

        row, col = position
        # Spiral score: combine radius and angle
        radius = self._radius_cache[row, col]
        angle = self._angle_cache[row, col]

        # Lower radius and angle = visited earlier
        score = -(radius * 10 + angle)
        return score

    def get_name(self) -> str:
        return "Spiral Walk"


class EdgeFollowingWalk(WalkStrategy):
    """Follow edges/boundaries detected by Canny."""

    def __init__(self, edge_strength_threshold: float = 0.5):
        self.threshold = edge_strength_threshold
        self._edge_cache = None

    def compute_score(
        self,
        image: np.ndarray,
        position: tuple[int, int],
        visited: set,
        walk_history: list[WalkStep],
    ) -> float:
        if self._edge_cache is None:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image.astype(np.uint8)

            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            self._edge_cache = edges.astype(float) / 255.0

        row, col = position
        return self._edge_cache[row, col]

    def get_name(self) -> str:
        return "Edge-Following Walk"


class ContourWigglingWalk(WalkStrategy):
    """
    Follows strong contours/edges while wiggling across both sides.
    Oscillates perpendicular to edge direction to explore edge boundaries.
    """

    def __init__(self, wiggle_amplitude: int = 2):
        """
        Args:
            wiggle_amplitude: How far to wiggle perpendicular to edge (in pixels)
        """
        self.wiggle_amplitude = wiggle_amplitude
        self._edge_cache = None
        self._gradient_x = None
        self._gradient_y = None
        self._wiggle_side = 1  # Alternates between 1 and -1

    def compute_score(
        self,
        image: np.ndarray,
        position: tuple[int, int],
        visited: set,
        walk_history: list[WalkStep],
    ) -> float:
        if self._edge_cache is None:
            # Compute edges and gradients
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image.astype(np.uint8)

            # Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            self._edge_cache = edges.astype(float) / 255.0

            # Compute gradients for edge orientation
            self._gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            self._gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        row, col = position

        # Base score: prefer edge pixels
        edge_score = self._edge_cache[row, col]

        # Compute perpendicular direction to edge
        # Edge direction is perpendicular to gradient
        gx = self._gradient_x[row, col]
        gy = self._gradient_y[row, col]

        # Perpendicular direction (90 degrees rotated from gradient)
        perp_x = -gy
        perp_y = gx

        # Normalize
        magnitude = np.sqrt(perp_x**2 + perp_y**2) + 1e-8
        perp_x /= magnitude
        perp_y /= magnitude

        # Check if this position is on the "wiggle side"
        # Compare position relative to last position in walk history
        if walk_history:
            last_pos = walk_history[-1].position
            direction_to_here = (row - last_pos[0], col - last_pos[1])

            # Dot product with perpendicular direction
            dot_product = direction_to_here[0] * perp_y + direction_to_here[1] * perp_x

            # Prefer moving in the current wiggle direction
            wiggle_bonus = 1.0 if (dot_product * self._wiggle_side) > 0 else 0.5

            # Every few steps, flip the wiggle side
            if len(walk_history) % (self.wiggle_amplitude * 2) == 0:
                self._wiggle_side *= -1
        else:
            wiggle_bonus = 1.0

        return edge_score * wiggle_bonus

    def get_name(self) -> str:
        return "Contour-Wiggling Walk"


class NoBacktrackingMinChangeWalk(WalkStrategy):
    """
    Greedy walk in direction of smallest change possible.
    Never retraces steps - always moves to unvisited neighbor with minimum gradient.
    """

    def __init__(self):
        self._gradient_cache = None

    def compute_score(
        self,
        image: np.ndarray,
        position: tuple[int, int],
        visited: set,
        walk_history: list[WalkStep],
    ) -> float:
        if self._gradient_cache is None:
            if len(image.shape) == 3:
                gray = np.sum(image, axis=2)
            else:
                gray = image

            grad_x = ndimage.sobel(gray, axis=1)
            grad_y = ndimage.sobel(gray, axis=0)
            self._gradient_cache = np.sqrt(grad_x**2 + grad_y**2)

        row, col = position
        score = self._gradient_cache[row, col]
        # Negative score because we want minimum
        return -score

    def get_name(self) -> str:
        return "No-Backtracking Minimum Change Walk"


class RandomWalk(WalkStrategy):
    """Completely random walk (baseline)."""

    def compute_score(
        self,
        image: np.ndarray,
        position: tuple[int, int],
        visited: set,
        walk_history: list[WalkStep],
    ) -> float:
        return np.random.random()

    def get_name(self) -> str:
        return "Random Walk"


class ImageWalker:
    """Main walker class that executes walk strategies."""

    def __init__(self, image: np.ndarray):
        """
        Args:
            image: Input image (H, W, C) or (H, W)
        """
        self.image = image.copy()
        self.shape = image.shape[:2]  # (height, width)

    def walk(
        self,
        strategy: WalkStrategy,
        start_position: tuple[int, int] | None = None,
        max_steps: int | None = None,
        connectivity: int = 8,
    ) -> list[WalkStep]:
        """
        Execute a walk through the image.

        Args:
            strategy: Walk strategy to use
            start_position: Starting position (row, col). If None, start at center.
            max_steps: Maximum number of steps. If None, visit all pixels.
            connectivity: 4 or 8 connected neighbors

        Returns:
            List of WalkStep objects representing the path
        """
        h, w = self.shape

        if start_position is None:
            start_position = (h // 2, w // 2)

        if max_steps is None:
            max_steps = h * w

        visited = set()
        walk_history = []
        current = start_position

        for _step_num in range(max_steps):
            # Record current step
            row, col = current
            pixel_value = (
                self.image[row, col] if len(self.image.shape) == 3 else self.image[row, col]
            )

            step = WalkStep(
                position=current,
                value=pixel_value,
                gradient_magnitude=0.0,  # Could compute if needed
                direction=None,
            )
            walk_history.append(step)
            visited.add(current)

            # Get neighbors
            neighbors = strategy.get_neighbors(current, self.shape, visited, connectivity)

            if not neighbors:
                # No unvisited neighbors, try to jump to unvisited pixel
                unvisited = [(i, j) for i in range(h) for j in range(w) if (i, j) not in visited]
                if not unvisited:
                    break  # All pixels visited

                # Jump to best unvisited pixel
                scores = [
                    strategy.compute_score(self.image, pos, visited, walk_history)
                    for pos in unvisited
                ]
                current = unvisited[np.argmax(scores)]
            else:
                # Score all neighbors
                scores = [
                    strategy.compute_score(self.image, pos, visited, walk_history)
                    for pos in neighbors
                ]

                # Select best neighbor
                best_idx = np.argmax(scores)
                next_pos = neighbors[best_idx]

                # Record direction
                step.direction = (next_pos[0] - current[0], next_pos[1] - current[1])

                current = next_pos

        return walk_history

    def visualize(
        self,
        walk_path: list[WalkStep],
        line_thickness: int = 1,
        color: tuple[int, int, int] = (255, 0, 0),
        show_start: bool = True,
        show_end: bool = True,
    ) -> np.ndarray:
        """
        Visualize walk path on image.

        Args:
            walk_path: Path from walk()
            line_thickness: Thickness of path line
            color: RGB color for path
            show_start: Mark start position
            show_end: Mark end position

        Returns:
            Image with path drawn
        """
        # Create output image
        if len(self.image.shape) == 2:
            output = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            output = self.image.astype(np.uint8).copy()

        # Draw path
        for i in range(len(walk_path) - 1):
            pt1 = (walk_path[i].position[1], walk_path[i].position[0])  # (col, row) for cv2
            pt2 = (walk_path[i + 1].position[1], walk_path[i + 1].position[0])
            cv2.line(output, pt1, pt2, color, line_thickness)

        # Mark start
        if show_start and len(walk_path) > 0:
            start = (walk_path[0].position[1], walk_path[0].position[0])
            cv2.circle(output, start, 5, (0, 255, 0), -1)  # Green circle

        # Mark end
        if show_end and len(walk_path) > 0:
            end = (walk_path[-1].position[1], walk_path[-1].position[0])
            cv2.circle(output, end, 5, (0, 0, 255), -1)  # Red circle

        return output
