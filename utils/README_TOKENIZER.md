# Walk Tokenizer - Image Dynamics as Sequences

## Overview

The Walk Tokenizer converts image walks into discrete token sequences for **next-token prediction**, the fundamental task of language models. Instead of predicting absolute pixel values, we predict the **CHANGE**:

- **Direction**: Which way will the walk move next? (8 directions + terminate)
- **Delta**: How much will the pixel value change?

This focuses on **image dynamics** rather than image state - analogous to predicting velocities instead of positions.

## Why Predict Change?

Predicting change (derivatives) rather than absolute values has several advantages:

1. **Invariance**: Changes are more invariant to global brightness/contrast shifts
2. **Compression**: Deltas typically have lower entropy than absolute values
3. **Semantics**: Changes correspond to edges, boundaries, and texture - the meaningful visual features
4. **Alignment with NTP**: Natural language predicts the next word (a transition), not a complete sentence state

## Quick Start

```python
from utils.image_walker import ImageWalker, BrightnessGradientWalk
from utils.walk_tokenizer import WalkTokenizer
import cv2

# Load image and execute walk
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

walker = ImageWalker(image)
path = walker.walk(BrightnessGradientWalk(maximize=True), max_steps=500)

# Tokenize the walk
tokenizer = WalkTokenizer(n_value_bins=64, value_range=(-255, 255))
tokens = tokenizer.tokenize_walk(path)

# Each token contains:
#   - direction: Direction enum (RIGHT, DOWN, UP_LEFT, etc.)
#   - value_delta: Change in pixel value (RGB array or scalar)
#   - position: Where the walk moved to
#   - step_index: Index in the walk sequence

print(f"Generated {len(tokens)} tokens")
for i, token in enumerate(tokens[:5]):
    print(f"Step {i}: {token.direction.name} -> Δ={token.value_delta}")
```

## Token Format

### Direction Token

8-connected directions plus termination:

```
Direction.RIGHT        (0, 1)   →
Direction.DOWN_RIGHT   (1, 1)   ↘
Direction.DOWN         (1, 0)   ↓
Direction.DOWN_LEFT    (1, -1)  ↙
Direction.LEFT         (0, -1)  ←
Direction.UP_LEFT      (-1, -1) ↖
Direction.UP           (-1, 0)  ↑
Direction.UP_RIGHT     (-1, 1)  ↗
Direction.TERMINATE    -        (walk ends)
```

### Value Delta Token

Continuous or discretized change in pixel value:

```python
# Continuous (for analysis)
token.value_delta  # [-255, 255] for grayscale, (3,) array for RGB

# Discretized (for training)
binned_delta = tokenizer.discretize_value_delta(token.value_delta)
# Returns bin index [0, n_value_bins-1]
```

## Creating Training Datasets

```python
# Create next-token prediction dataset
dataset = tokenizer.create_prediction_dataset(path, context_length=8)

# Each sample contains:
for sample in dataset:
    context_dirs = sample['context_directions']      # List[int] - last 8 directions
    context_deltas = sample['context_deltas']        # List[int] - last 8 binned deltas
    target_dir = sample['target_direction']          # int - direction to predict
    target_delta = sample['target_delta']            # int - delta to predict
    position = sample['position']                    # (row, col) - for analysis
```

## Use Cases

### 1. Self-Supervised Pretraining

Train a transformer to predict next (direction, delta) tokens:

```python
from utils.walk_tokenizer import WalkTokenizer
from utils.image_walker import ImageWalker, BrightnessGradientWalk

# Process a dataset of images
all_samples = []

for image in dataset:
    walker = ImageWalker(image)

    # Use multiple walk strategies for diversity
    for strategy in [BrightnessGradientWalk(), SaliencyWalk(), EdgeFollowingWalk()]:
        path = walker.walk(strategy, max_steps=1000)
        samples = tokenizer.create_prediction_dataset(path, context_length=16)
        all_samples.extend(samples)

# Train autoregressive model
# Input: [dir_1, ..., dir_t, delta_1, ..., delta_t]
# Output: (dir_t+1, delta_t+1)
```

### 2. Representation Learning

The learned direction/delta embeddings capture:
- Edge orientations (direction preferences)
- Texture patterns (delta distributions)
- Object boundaries (direction changes)
- Local structure (context sequences)

### 3. Analyzing Walk Strategies

```python
# Compare different walk strategies
stats = tokenizer.get_statistics(path)

print(f"Direction distribution: {stats['direction_distribution']}")
print(f"Mean delta: {stats['delta_mean']}")
print(f"Std delta: {stats['delta_std']}")

# Gradient walks prefer certain directions (along edges)
# Random walks have uniform direction distribution
# Saliency walks have high-magnitude deltas
```

## Visualization

```python
from utils.walk_tokenizer import visualize_token_sequence

viz = visualize_token_sequence(tokens, image, output_path='tokens.png')

# Arrows colored by direction
# Arrow thickness indicates magnitude of value change
# Green dot = start, Red dot = end
```

## Advanced: Multi-Scale Tokenization

Combine pixel-level walks with superpixel-level walks:

```python
from utils.superpixel_walker import SuperpixelWalker

# Pixel-level tokens (fine-grained)
pixel_path = ImageWalker(image).walk(BrightnessGradientWalk(), max_steps=500)
pixel_tokens = tokenizer.tokenize_walk(pixel_path)

# Superpixel-level tokens (coarse-grained)
sp_walker = SuperpixelWalker(image, n_segments=50)
sp_order = sp_walker.walk_by_gradient(maximize=True)

# Hierarchical tokenization: superpixel transitions + pixel transitions within each
```

## Token Statistics

For typical natural images with gradient-based walks:

- **Direction entropy**: ~2.0-2.5 bits (not uniform - prefer edge-aligned directions)
- **Delta entropy**: ~4-5 bits (depends on n_value_bins)
- **Sequence length**: 100-1000 tokens before natural termination
- **Compression ratio**: 8-10x vs. raster scan (due to coherent paths)

## Design Rationale

### Why Direction + Delta?

This factorization separates:
- **Spatial structure** (direction): Where edges/boundaries go
- **Intensity structure** (delta): How bright/colorful regions transition

Both are essential for visual understanding and easier to learn separately than a combined token space of size 9 × n_value_bins.

### Why Allow Natural Termination?

Walks that naturally terminate create **semantically coherent sequences**:
- Edge-following walks trace object boundaries
- Gradient walks follow intensity ridges/valleys
- Each terminated sequence is a "visual phrase"

For full coverage, multiple walks can be combined - but individual walks should be meaningful units.

## Implementation Notes

- **Grayscale vs RGB**: Deltas are scalars for grayscale, (3,) arrays for RGB
- **Discretization**: Configurable bins with adjustable range
- **Direction encoding**: IntEnum for type safety and readability
- **Context windows**: Typically 8-16 tokens for local prediction
- **Computational cost**: O(n) where n = path length (very fast)

## Examples

See `experiments/demo_walk_tokenization.py` for complete demonstrations:

```bash
python experiments/demo_walk_tokenization.py
```

This generates:
- `token_visualization.png` - Arrows showing directions and magnitudes
- `direction_analysis.png` - Direction distributions across strategies
- `delta_distributions.png` - Value change histograms
- Console output with tokenization statistics
