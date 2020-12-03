# pylint:disable=missing-docstring
import logging
import typing as ta

import torch
from gym.spaces import Box
from raylab.policy.losses import Loss
from raylab.policy.losses import MaximumLikelihood
from raylab.policy.model_based.lightning import LightningModel as BaseModel
from raylab.policy.modules.critic import VValue
from raylab.policy.modules.model import build_single as build_standard
from raylab.policy.modules.model import SingleSpec as StandardSpec
from raylab.policy.modules.model import StochasticModel
from raylab.torch.utils import convert_to_tensor
from raylab.utils.types import StatDict
from raylab.utils.types import TensorDict
from torch import Tensor

from vmac.policy.losses import VAML
from vmac.policy.modules.model.stochastic.single import build_single as build_fixed
from vmac.policy.modules.model.stochastic.single import Spec as FixedSpec

__all__ = [
    "MaximumLikelihood",
    "VValue",
    "build_standard",
    "StandardSpec",
    "VAML",
    "build_fixed",
    "FixedSpec",
]

logger = logging.getLogger(__name__)


# ======================================================================================
# BUILDER
# ======================================================================================


def build_single(
    obs_space: Box, action_space: Box, spec: ta.Union[FixedSpec, StandardSpec]
) -> StochasticModel:
    if isinstance(spec, FixedSpec):
        return build_fixed(obs_space, action_space, spec)
    return build_standard(obs_space, action_space, spec)


# ======================================================================================
# BASE MODEL
# ======================================================================================


class LightningModel(BaseModel):
    # pylint:disable=too-many-ancestors,arguments-differ
    def __init__(self, model: StochasticModel, loss: Loss):
        super().__init__(model, loss, None)

        self.hparams.learning_rate = 1e-3
        self.hparams.weight_decay = 0

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def configure_losses(self, loss: Loss):
        self.train_loss = loss

        def val_loss(batch: TensorDict) -> ta.Tuple[Tensor, StatDict]:
            tensor, stat = loss(batch)
            if "loss" in stat:
                tensor = torch.full_like(tensor, fill_value=stat["loss"])
            return tensor, stat

        self.val_loss = self.test_loss = val_loss
