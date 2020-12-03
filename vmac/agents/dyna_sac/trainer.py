"""Soft Actor-Critic with Dyna-like data augmentation for critic learning."""
from raylab.agents.model_based import ModelBasedMixin
from raylab.agents.sac import SACTrainer

from .policy import DynaSACTorchPolicy


class DynaSACTrainer(ModelBasedMixin, SACTrainer):
    """Single agent trainer for Dyna-SAC."""

    # pylint:disable=abstract-method

    _name = "Dyna-SAC"
    _policy_class = DynaSACTorchPolicy
