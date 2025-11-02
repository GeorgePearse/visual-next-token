"""
RL-based Image Navigation

Curiosity-driven reinforcement learning for learning to navigate images
to information-dense regions. Agent learns paths that maximize prediction
error (not accuracy), forcing exploration of semantically informative areas.

Core components:
- SemanticEncoder: DINOv2-based feature extraction
- NavigationPolicy: PPO policy for path selection
- ForwardDynamicsModel: Predict future features from current state + action
- ImageNavigationEnv: MDP environment for image traversal
- RLTrainer: Two-phase training (frozen encoder → fine-tuned encoder)

Extensions (Optional):
- ExtendedActionSpace: Jump/scout actions for long-range exploration
- HierarchicalPolicy: Two-level policy for extended actions
- ScoutingRewardModifier: Reward adjustments for scout actions
"""

from .encoder import SemanticEncoder
from .environment import ImageNavigationEnv
from .extensions import (
    ExtendedActionSpace,
    HierarchicalPolicy,
    ScoutingRewardModifier,
)
from .forward_dynamics import ForwardDynamicsModel, RNDIntrinsicMotivation
from .policy import NavigationPolicy
from .trainer import RLTrainer

__all__ = [
    "SemanticEncoder",
    "ImageNavigationEnv",
    "ForwardDynamicsModel",
    "RNDIntrinsicMotivation",
    "NavigationPolicy",
    "RLTrainer",
    # Extensions
    "ExtendedActionSpace",
    "HierarchicalPolicy",
    "ScoutingRewardModifier",
]
