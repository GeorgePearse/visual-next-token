"""
Walk Tokenizer - Convert image walks into discrete tokens for prediction.

Instead of predicting absolute pixel values, we predict the CHANGE:
- Direction: Which way will the walk move next? (8 directions + terminate)
- Delta: What's the change in pixel value from current position?

This aligns with "next token prediction" by focusing on dynamics rather than state.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from utils.image_walker import WalkStep


class Direction(IntEnum):
    """8-connected directions plus termination."""

    RIGHT = 0  # (0, 1)
    DOWN_RIGHT = 1  # (1, 1)
    DOWN = 2  # (1, 0)
    DOWN_LEFT = 3  # (1, -1)
    LEFT = 4  # (0, -1)
    UP_LEFT = 5  # (-1, -1)
    UP = 6  # (-1, 0)
    UP_RIGHT = 7  # (-1, 1)
    TERMINATE = 8  # Walk ends


# Mapping from (dr, dc) to Direction
DELTA_TO_DIRECTION = {
    (0, 1): Direction.RIGHT,
    (1, 1): Direction.DOWN_RIGHT,
    (1, 0): Direction.DOWN,
    (1, -1): Direction.DOWN_LEFT,
    (0, -1): Direction.LEFT,
    (-1, -1): Direction.UP_LEFT,
    (-1, 0): Direction.UP,
    (-1, 1): Direction.UP_RIGHT,
}

DIRECTION_TO_DELTA = {v: k for k, v in DELTA_TO_DIRECTION.items()}


@dataclass
class WalkToken:
    """
    A single token in the walk sequence.

    Represents the transition from one step to the next:
    - direction: Which way did we move?
    - value_delta: How much did the pixel value change?
    """

    direction: Direction
    value_delta: float | np.ndarray  # Scalar for grayscale, (3,) array for RGB
    position: tuple[int, int]  # Position we moved TO
    step_index: int  # Index in the walk sequence


class WalkTokenizer:
    """
    Converts image walks into discrete tokens for prediction tasks.

    Focus on CHANGE rather than absolute values:
    - Predict next direction of movement
    - Predict change in pixel value (delta)
    """

    def __init__(
        self,
        n_value_bins: int = 64,
        value_range: tuple[float, float] = (-255.0, 255.0),
    ):
        """
        Args:
            n_value_bins: Number of bins for discretizing value changes
            value_range: Min/max range for value deltas
        """
        self.n_value_bins = n_value_bins
        self.value_range = value_range
        self.bin_edges = np.linspace(value_range[0], value_range[1], n_value_bins + 1)

    def tokenize_walk(self, walk_path: list[WalkStep]) -> list[WalkToken]:
        """
        Convert a walk path into a sequence of tokens.

        Each token represents the transition FROM step[i] TO step[i+1].

        Args:
            walk_path: Sequence of walk steps

        Returns:
            List of WalkTokens (length = len(walk_path) - 1)
        """
        if len(walk_path) < 2:
            return []

        tokens = []

        for i in range(len(walk_path) - 1):
            current_step = walk_path[i]
            next_step = walk_path[i + 1]

            # Compute direction
            dr = next_step.position[0] - current_step.position[0]
            dc = next_step.position[1] - current_step.position[1]

            direction = DELTA_TO_DIRECTION.get((dr, dc), Direction.TERMINATE)

            # Compute value delta
            current_val = current_step.value
            next_val = next_step.value

            if isinstance(current_val, np.ndarray):
                # RGB image
                value_delta = next_val.astype(float) - current_val.astype(float)
            else:
                # Grayscale
                value_delta = float(next_val) - float(current_val)

            tokens.append(
                WalkToken(
                    direction=direction,
                    value_delta=value_delta,
                    position=next_step.position,
                    step_index=i + 1,
                )
            )

        return tokens

    def discretize_value_delta(self, delta: float | np.ndarray) -> int | np.ndarray:
        """
        Discretize continuous value delta into bins.

        Args:
            delta: Continuous value change (scalar or RGB array)

        Returns:
            Bin index (int for grayscale, array for RGB)
        """
        if isinstance(delta, np.ndarray):
            # RGB - discretize each channel
            return np.digitize(delta, self.bin_edges) - 1
        else:
            # Grayscale
            return int(np.digitize(delta, self.bin_edges) - 1)

    def get_discrete_tokens(self, walk_path: list[WalkStep]) -> list[tuple[int, int | np.ndarray]]:
        """
        Get fully discrete tokens for training.

        Returns:
            List of (direction_id, binned_delta) tuples
        """
        tokens = self.tokenize_walk(walk_path)

        discrete = []
        for token in tokens:
            direction_id = int(token.direction)
            binned_delta = self.discretize_value_delta(token.value_delta)
            discrete.append((direction_id, binned_delta))

        return discrete

    def create_prediction_dataset(
        self, walk_path: list[WalkStep], context_length: int = 8
    ) -> list[dict]:
        """
        Create dataset for next-token prediction.

        Each sample has:
        - context: Last N tokens (directions + deltas)
        - target: Next token to predict

        Args:
            walk_path: Walk sequence
            context_length: Number of previous tokens to use as context

        Returns:
            List of training samples
        """
        tokens = self.tokenize_walk(walk_path)
        discrete_tokens = self.get_discrete_tokens(walk_path)

        if len(tokens) < context_length + 1:
            return []

        dataset = []

        for i in range(context_length, len(discrete_tokens)):
            # Context: previous tokens
            context = discrete_tokens[i - context_length : i]

            # Target: next token
            target = discrete_tokens[i]

            dataset.append(
                {
                    "context_directions": [t[0] for t in context],
                    "context_deltas": [t[1] for t in context],
                    "target_direction": target[0],
                    "target_delta": target[1],
                    "position": tokens[i].position,
                }
            )

        return dataset

    def get_statistics(self, walk_path: list[WalkStep]) -> dict:
        """
        Compute statistics about a walk for analysis.

        Returns:
            Dictionary with:
            - direction_distribution: Count of each direction
            - delta_mean: Mean value change
            - delta_std: Std of value changes
            - total_steps: Number of transitions
        """
        tokens = self.tokenize_walk(walk_path)

        if not tokens:
            return {
                "direction_distribution": {},
                "delta_mean": 0.0,
                "delta_std": 0.0,
                "total_steps": 0,
            }

        # Direction distribution
        direction_counts = {}
        for token in tokens:
            direction_counts[token.direction] = direction_counts.get(token.direction, 0) + 1

        # Value delta statistics
        deltas = [token.value_delta for token in tokens]

        if isinstance(deltas[0], np.ndarray):
            # RGB - compute per-channel stats
            deltas_array = np.array(deltas)
            delta_mean = deltas_array.mean(axis=0)
            delta_std = deltas_array.std(axis=0)
        else:
            # Grayscale
            delta_mean = float(np.mean(deltas))
            delta_std = float(np.std(deltas))

        return {
            "direction_distribution": direction_counts,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
            "total_steps": len(tokens),
        }


def visualize_token_sequence(
    tokens: list[WalkToken], image: np.ndarray, output_path: str | None = None
) -> np.ndarray:
    """
    Visualize a token sequence showing directions and value changes.

    Args:
        tokens: Sequence of walk tokens
        image: Original image
        output_path: Optional path to save visualization

    Returns:
        Visualization image
    """
    import cv2

    viz = image.copy()

    # Direction colors (HSV mapped to 8 directions)
    direction_colors = {
        Direction.RIGHT: (255, 0, 0),  # Red
        Direction.DOWN_RIGHT: (255, 128, 0),  # Orange
        Direction.DOWN: (255, 255, 0),  # Yellow
        Direction.DOWN_LEFT: (128, 255, 0),  # Yellow-green
        Direction.LEFT: (0, 255, 0),  # Green
        Direction.UP_LEFT: (0, 255, 255),  # Cyan
        Direction.UP: (0, 128, 255),  # Light blue
        Direction.UP_RIGHT: (128, 0, 255),  # Purple
        Direction.TERMINATE: (128, 128, 128),  # Gray
    }

    # Draw arrows for each token
    for i, token in enumerate(tokens):
        if i == 0:
            continue  # Skip first (no previous position)

        prev_token = tokens[i - 1]

        # Arrow from previous to current position
        pt1 = (prev_token.position[1], prev_token.position[0])  # (col, row)
        pt2 = (token.position[1], token.position[0])

        color = direction_colors.get(token.direction, (255, 255, 255))

        # Arrow thickness based on magnitude of value change
        if isinstance(token.value_delta, np.ndarray):
            magnitude = float(np.linalg.norm(token.value_delta))
        else:
            magnitude = abs(token.value_delta)

        thickness = max(1, int(magnitude / 50))

        cv2.arrowedLine(viz, pt1, pt2, color, thickness, tipLength=0.3)

    # Mark start and end
    if tokens:
        start_pos = (tokens[0].position[1], tokens[0].position[0])
        end_pos = (tokens[-1].position[1], tokens[-1].position[0])

        cv2.circle(viz, start_pos, 5, (0, 255, 0), -1)  # Green start
        cv2.circle(viz, end_pos, 5, (0, 0, 255), -1)  # Red end

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(viz, cv2.COLOR_RGB2BGR))

    return viz
