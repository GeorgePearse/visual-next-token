# Quick Start

Get started with RL-based image navigation in 5 minutes!

## 1. Train on Test Image

The fastest way to see the system in action:

```bash
python experiments/train_rl_navigator.py --config quick_test
```

This uses:
- Synthetic test image (colored shapes on gradient)
- 100 episodes Phase 1 + 50 episodes Phase 2
- Small model (`dinov2_vits14`)
- Should complete in ~10 minutes on GPU

## 2. Train on Your Own Image

```bash
python experiments/train_rl_navigator.py \
    --image path/to/your/image.jpg \
    --config quick_test
```

The image will be automatically resized to 512×512 if larger.

## 3. Visualize Learned Paths

After training completes:

```bash
python experiments/visualize_rl_paths.py \
    --checkpoint checkpoints/rl_navigator/final.pt \
    --n_episodes 5
```

This will:
- Load the trained policy
- Run 5 episodes on the test image
- Save visualizations to `experiments/outputs/rl_paths/`
- Display a comparison of the learned paths

## Understanding the Output

### Training Logs

```
Phase 1 - Episode 10/100
  Avg Reward: 12.34
  Avg Length: 234.5
  Coverage: 45.23%
  Pred Error: 0.0234
  Policy Loss: 0.1234
  Value Loss: 0.0456
  Entropy: 1.234
```

| Metric | Meaning |
|--------|---------|
| **Avg Reward** | Intrinsic reward (prediction error + coverage bonus) |
| **Avg Length** | Steps per episode (max 500) |
| **Coverage** | % of image visited |
| **Pred Error** | Forward model prediction error |
| **Policy Loss** | PPO policy loss |
| **Value Loss** | Value function MSE |
| **Entropy** | Policy entropy (exploration) |

### Visualizations

The visualization script creates:

- `rl_path_ep1.png` - Individual episode paths
- `comparison.png` - Side-by-side comparison of first 4 episodes

Paths show:
- **Green circle**: Start position
- **Red circle**: End position
- **Colored line**: Path taken

## Configuration Presets

| Config | Phase 1 | Phase 2 | Use Case |
|--------|---------|---------|----------|
| `quick_test` | 100 | 50 | Testing/debugging |
| `default` | 10000 | 5000 | Standard training |
| `rnd` | 10000 | 5000 | Use RND instead of ICM |
| `long` | 20000 | 10000 | Extended training with larger model |

## Using Different Configurations

### RND (Random Network Distillation)

```bash
python experiments/train_rl_navigator.py --config rnd
```

RND is an alternative to forward dynamics:
- More stable training
- No action conditioning needed
- Good for complex state spaces

### Long Training with Larger Model

```bash
python experiments/train_rl_navigator.py --config long
```

Uses:
- `dinov2_vitb14` (768-dim features vs 384-dim)
- 30k total episodes (20k + 10k)
- Better final performance, but slower

## Custom Configuration

For full control, use Python API:

```python
from techniques.rl_navigation import RLTrainer
import cv2

# Load your image
image = cv2.imread("my_image.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create trainer with custom config
trainer = RLTrainer(
    image=image,
    encoder_name="dinov2_vits14",
    phase1_episodes=5000,
    phase2_episodes=2000,
    policy_lr=1e-4,
    predictor_lr=5e-4,
    use_rnd=False,
    reward_lambda=0.1,
    coverage_bonus_weight=0.05,
    save_dir="my_checkpoints",
)

# Train
trainer.train()
```

## Resuming from Checkpoint

```bash
python experiments/train_rl_navigator.py \
    --resume checkpoints/rl_navigator/phase1_ep1000.pt
```

## Common Issues

### Out of Memory

If you get CUDA OOM errors:

1. Use smaller model: `dinov2_vits14` instead of `vitb14`
2. Reduce rollout steps in config
3. Use CPU: `--device cpu` (slower)

### Slow Training

- Ensure GPU is being used: check `Device: cuda` in output
- Use `quick_test` config first
- Consider reducing image size in training script

### Low Coverage

If the agent doesn't explore much:

- Increase `coverage_bonus_weight` (default 0.1)
- Check that prediction error is non-zero
- Verify entropy is > 0 (policy is exploring)

## Next Steps

- [Architecture Deep Dive](../techniques/rl-navigation/architecture.md) - Understand how it works
- [Training Guide](../techniques/rl-navigation/training.md) - Advanced training strategies
- [Extensions](../techniques/rl-navigation/extensions.md) - Jump/scout actions
