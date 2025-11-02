# Project TODO List

This page tracks the development status of Visual Next Token, including completed features, ongoing work, and planned improvements.

**Last Updated**: 2025-11-02

---

## ✅ Completed

### Core RL Navigation Implementation
- [x] Semantic encoder wrapper (DINOv2) with freeze/unfreeze capabilities
- [x] Image navigation environment (MDP formulation)
- [x] Forward dynamics model (ICM) for intrinsic motivation
- [x] Random Network Distillation (RND) as alternative intrinsic motivation
- [x] PPO policy with actor-critic architecture
- [x] GAE (Generalized Advantage Estimation) for advantage computation
- [x] Two-phase training system (frozen → fine-tuned encoder)
- [x] Configuration management with multiple presets
- [x] Training script with checkpoint/resume support
- [x] Visualization script for learned paths
- [x] Extended action spaces (jump/scout actions)

### Documentation
- [x] MkDocs Material documentation site
- [x] Comprehensive README with quick start
- [x] Installation guide
- [x] Quick start tutorial
- [x] RL navigation architecture documentation
- [x] Research paper summaries (ICM, RND, PPO, DINOv2, GAE)
- [x] References directory with detailed paper notes
- [x] GitHub Pages deployment
- [x] Conceptual framing clarified (rolling-window accuracy)

### Repository Setup
- [x] Repository renamed to `visual-next-token`
- [x] Git repository initialized and pushed
- [x] GitHub Pages enabled
- [x] Documentation deployed

---

## 🚧 In Progress

### Documentation Improvements
- [x] Complete architecture deep-dive page (techniques/rl_navigation/architecture.md)
- [x] Training guide with best practices (techniques/rl_navigation/training.md)
- [x] Extensions guide for jump/scout actions (techniques/rl_navigation/extensions.md)
- [x] API reference documentation (api/rl_navigation.md)
- [x] Contributing guide (development/contributing.md)

### Code Quality
- [ ] Add type hints throughout codebase
- [ ] Add docstrings to all public methods
- [ ] Set up pytest test suite
- [ ] Add unit tests for core components
- [ ] Add integration tests for training pipeline

---

## 📋 High Priority TODO

### Critical Features

#### 1. Reward Mechanism Alignment ✅ **FIXED** (2025-11-02)
**Status**: Complete - code now matches documentation
**Priority**: High (RESOLVED)
**Description**: Ensured reward computation correctly implements rolling-window prediction accuracy.

**Issue Found**:
The code was rewarding PREDICTION ERROR (high error = high reward), which contradicts the documented rolling-window accuracy framing. This encouraged seeking surprising/chaotic regions rather than semantically coherent paths.

**Previous implementation**:
```python
# environment.py:_compute_reward (OLD)
prediction_error = torch.sum((predicted - actual) ** 2).item()
total_reward = prediction_error + lookahead_reward + coverage_bonus  # ❌ Maximizes error!
```

**Fixed implementation**:
```python
# environment.py:_compute_reward (NEW)
prediction_error = torch.sum((predicted - actual) ** 2).item()
immediate_accuracy_reward = -prediction_error  # ✅ Negate: accuracy = -error
lookahead_accuracy_reward = -lookahead_error   # ✅ Negate lookahead too
total_reward = immediate_accuracy_reward + lookahead_accuracy_reward + coverage_bonus
```

**Changes made**:
- [x] ~~Verified lookahead reward implements rolling-window accuracy~~ → **FIXED**: Now negates error
- [x] ~~Inverted sign to make accuracy-based~~ → **DONE**: `reward = -error`
- [x] ~~Added comprehensive comments~~ → **DONE**: Documented rolling-window framing throughout
- [x] ~~Updated variable names~~ → **DONE**: `immediate_accuracy_reward`, `lookahead_accuracy_reward`
- [x] Renamed `_compute_lookahead_reward()` → `_compute_lookahead_error()` for clarity
- [x] Updated all docstrings in `environment.py` and `forward_dynamics.py`
- [x] Added car edge example in reward computation docstring

**Files modified**:
- `techniques/rl_navigation/environment.py` (lines 1-325)
- `techniques/rl_navigation/forward_dynamics.py` (lines 1-220)

**Impact**:
This is a **CRITICAL** fix that fundamentally changes agent behavior:
- **OLD**: Agent seeks high-error (surprising) transitions → chaotic/noisy regions
- **NEW**: Agent seeks low-error (predictable) paths → semantically coherent regions

**Next steps**:
- Run experiments with fixed reward to validate behavior change
- Compare learned paths before/after fix (expect more semantic coherence)

#### 2. Testing Infrastructure
**Status**: Not started
**Priority**: High
**Description**: Add comprehensive test suite before making significant changes.

**Action items**:
- [ ] Set up pytest configuration
- [ ] Add tests for encoder (freeze/unfreeze, feature extraction)
- [ ] Add tests for environment (state transitions, reward computation)
- [ ] Add tests for policy (action sampling, evaluation)
- [ ] Add tests for forward dynamics (prediction, intrinsic reward)
- [ ] Add integration test for full training loop (quick config)
- [ ] Set up CI/CD with GitHub Actions

#### 3. Experiment Validation
**Status**: Not started
**Priority**: High
**Description**: Run experiments to validate that the system works as intended.

**Action items**:
- [ ] Run quick_test config on synthetic image
- [ ] Verify agent explores semantic regions (not random walk)
- [ ] Measure coverage and path statistics
- [ ] Compare RND vs ICM on same image
- [ ] Test on real-world images (ImageNet samples)
- [ ] Document experimental results

---

## 📊 Medium Priority TODO

### Feature Enhancements

#### 1. Advanced Visualization
**Files**: `experiments/visualize_rl_paths.py`

- [ ] Add heatmap of visited regions
- [ ] Show prediction error/accuracy over trajectory
- [ ] Visualize semantic feature space (t-SNE/UMAP)
- [ ] Animate path exploration over time
- [ ] Compare learned paths to baselines (random, greedy)

#### 2. Training Improvements
**Files**: `techniques/rl_navigation/trainer.py`

- [ ] Add TensorBoard logging
- [ ] Implement early stopping based on coverage
- [ ] Add learning rate scheduling
- [ ] Support distributed training (multi-GPU)
- [ ] Add curriculum learning (start with simple images)

#### 3. Environment Extensions
**Files**: `techniques/rl_navigation/environment.py`, `techniques/rl_navigation/extensions.py`

- [ ] Support multi-image training (generalization)
- [ ] Add image augmentation during training
- [ ] Implement hierarchical navigation (coarse → fine)
- [ ] Add semantic segmentation integration (ground truth comparison)
- [ ] Support different patch sizes (adaptive resolution)

#### 4. Documentation Pages
**Directory**: `docs/`

- [ ] Create architecture deep-dive (with diagrams)
- [ ] Write training guide with hyperparameter tuning tips
- [ ] Document extensions (jump/scout) with usage examples
- [ ] Add API reference (auto-generated from docstrings)
- [ ] Create troubleshooting guide
- [ ] Add gallery of learned paths on different images

---

## 🔬 Research & Experiments TODO

### Novel Research Directions

#### 1. Multi-Image Generalization
**Description**: Train on multiple images, test generalization to unseen images.

- [ ] Implement multi-image dataset loader
- [ ] Modify trainer to sample images per episode
- [ ] Test on held-out validation images
- [ ] Compare to single-image overfitting

#### 2. Semantic Segmentation Integration
**Description**: Use segmentation masks to evaluate semantic exploration quality.

- [ ] Integrate segmentation model (e.g., SAM)
- [ ] Measure coverage of different semantic classes
- [ ] Evaluate if agent prioritizes rare objects
- [ ] Compare to coverage-based baselines

#### 3. Ablation Studies
**Description**: Understand what components are critical.

**Experiments**:
- [ ] Frozen vs fine-tuned encoder (Phase 1 only vs Phase 2)
- [ ] ICM vs RND intrinsic motivation
- [ ] Different DINOv2 model sizes (vits14 vs vitb14 vs vitl14)
- [ ] Coverage bonus weight (0.0 vs 0.1 vs 0.5)
- [ ] Reward horizon (5 vs 10 vs 20 steps)
- [ ] Policy architecture (hidden dim, layers)

#### 4. Comparison to Baselines
**Description**: Establish that RL approach outperforms simpler alternatives.

**Baselines**:
- [ ] Random walk
- [ ] Greedy (always move to highest-error neighbor)
- [ ] Saliency-based navigation
- [ ] Optical flow following
- [ ] Edge-following heuristic

---

## 🐛 Known Issues

### Critical Bugs
_None currently identified_

### Minor Issues

1. ~~**MkDocs warnings for missing pages**~~ ✅ RESOLVED
   - ~~Several linked pages in navigation don't exist yet~~
   - ~~Warnings about broken internal links~~
   - **Fixed**: Created all missing documentation pages (2025-11-02)

2. **Extension tests not implemented**
   - `techniques/rl_navigation/extensions.py` has test code but needs PyTorch installed
   - **Fix**: Add to test suite with proper dependencies

3. ~~**Documentation links inconsistency**~~ ✅ RESOLVED
   - ~~Some use `rl-navigation` (kebab-case), some use `rl_navigation` (snake_case)~~
   - **Fixed**: Standardized on snake_case throughout (2025-11-02)

---

## 💡 Ideas & Future Work

### Long-term Research Directions

#### 1. Transfer Learning
- Pre-train navigator on large image dataset
- Fine-tune on specific domains (medical, satellite)
- Investigate what navigation patterns transfer

#### 2. Multi-Agent Exploration
- Multiple agents exploring same image
- Communication between agents
- Collaborative coverage optimization

#### 3. Active Vision
- Agent controls camera in 3D environment
- Navigate real-world scenes (not just images)
- Integration with robotics simulators

#### 4. Hierarchical Semantic Navigation
- Learn high-level semantic goals ("find animals")
- Decompose into low-level navigation primitives
- Options framework for temporally extended actions

#### 5. Self-Supervised Pre-training
- Use learned navigation to generate training data
- Discover semantic segmentation from exploration
- Bootstrap better feature representations

---

## 📝 Documentation TODO

### Pages to Create

1. **techniques/rl_navigation/architecture.md**
   - System architecture diagram
   - Component interaction details
   - Data flow visualization
   - Mathematical formulations

2. **techniques/rl_navigation/training.md**
   - Step-by-step training walkthrough
   - Hyperparameter tuning guide
   - Common issues and solutions
   - Performance optimization tips

3. **techniques/rl_navigation/extensions.md**
   - Jump/scout action usage
   - Hierarchical policy details
   - Custom action space design
   - Integration examples

4. **api/rl-navigation.md**
   - Auto-generated API reference
   - Class documentation
   - Method signatures
   - Usage examples

5. **development/contributing.md**
   - Contribution guidelines
   - Code style guide
   - PR process
   - Development setup

### Documentation Improvements

- [ ] Add more code examples throughout
- [ ] Create interactive Jupyter notebooks
- [ ] Add video demonstrations of learned paths
- [ ] Improve math rendering (LaTeX)
- [ ] Add bibliography/citations page
- [ ] Create FAQ section

---

## 🎯 Milestones

### Milestone 1: Core Validation (Target: 1 week)
- [ ] Fix reward mechanism if needed
- [ ] Run basic experiments
- [ ] Validate semantic exploration behavior
- [ ] Document experimental results

### Milestone 2: Testing & Quality (Target: 2 weeks)
- [ ] Complete test suite
- [ ] Add CI/CD
- [ ] Type hints throughout
- [ ] Documentation complete

### Milestone 3: Research Publication (Target: 1-2 months)
- [ ] Run comprehensive experiments
- [ ] Compare to baselines
- [ ] Ablation studies
- [ ] Write paper draft

---

## 📞 Questions & Decisions Needed

### Open Questions

1. **Reward formulation**: Is current implementation truly rolling-window accuracy, or does it need adjustment?
   - Current: Exponentially-weighted sum of prediction errors
   - Target: Should we invert to accuracy? Add baseline comparison?

2. **Training stability**: Are there any known stability issues with current setup?
   - Two-phase training seems sound
   - Need empirical validation

3. **Evaluation metrics**: What metrics best capture "semantic exploration quality"?
   - Coverage alone insufficient
   - Need semantic diversity metrics
   - Consider segmentation-based evaluation

4. **Baseline comparisons**: Which baselines are most important?
   - Random walk (essential)
   - Greedy prediction error (natural comparison)
   - Saliency-based (vision literature)

### Design Decisions

1. **Should we support multi-image training by default?**
   - Pro: Better generalization
   - Con: More complex, slower training
   - **Decision**: Add as optional mode, keep single-image as default

2. **Visualization: real-time or post-hoc only?**
   - Real-time helps debugging but slows training
   - **Decision**: Post-hoc by default, optional real-time flag

3. **Testing: unit vs integration focus?**
   - Both needed, but which first?
   - **Decision**: Start with integration tests (full training), then unit tests

---

## 🔄 Changelog

### 2025-11-02 (Evening Session)
- ✅ **CRITICAL FIX: Reward mechanism alignment**
  - Fixed code to match documented rolling-window accuracy framing
  - Changed from maximizing error to maximizing accuracy (negated error)
  - Updated `environment.py`: `reward = -error` instead of `reward = error`
  - Renamed `_compute_lookahead_reward()` → `_compute_lookahead_error()`
  - Updated all docstrings and comments in `environment.py` and `forward_dynamics.py`
  - Added car edge example in reward computation documentation
  - This fundamentally changes agent behavior: seeks coherent paths, not chaotic regions
- ✅ **Added Python RL frameworks reference** to `docs/references/index.md`
  - Production frameworks: Stable-Baselines3, Ray RLlib, CleanRL, Tianshou
  - Specialized frameworks: Gymnasium, PettingZoo, Sample Factory
  - Integration examples showing how to use SB3 with our environment
  - Guidance on choosing the right framework

### 2025-11-02 (Earlier)
- ✅ Created comprehensive TODO list
- ✅ Identified reward mechanism review as high priority
- ✅ Documented completed RL navigation implementation
- ✅ Listed all pending documentation pages
- ✅ Added research directions and ablation study plans
- ✅ **Resolved all MkDocs warnings** - created 5 comprehensive documentation pages:
  - Architecture deep dive
  - Training guide
  - Extensions guide (jump/scout actions)
  - API reference
  - Contributing guidelines
- ✅ **Fixed link inconsistencies** - standardized on snake_case throughout
- ✅ Zero build warnings achieved

---

## How to Update This TODO

When you complete a task:
1. Move from `[ ]` to `[x]`
2. Update the date in Changelog
3. Add notes about implementation details if relevant

When you add a new task:
1. Choose appropriate priority section
2. Add clear description and action items
3. Link to relevant files/docs
4. Update Changelog with addition
