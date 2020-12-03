# pylint:disable=missing-module-docstring
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from raylab.policy.losses.mixins import EnvFunctionsMixin
from raylab.policy.losses.mixins import UniformModelPriorMixin
from raylab.policy.losses.q_learning import QLearningMixin
from raylab.policy.losses.utils import dist_params_stats
from raylab.policy.modules.critic import QValueEnsemble
from raylab.policy.modules.critic import VValue
from raylab.policy.modules.model import SME
from raylab.policy.modules.model import StochasticModel
from raylab.utils.types import StatDict
from raylab.utils.types import TensorDict
from torch import Tensor

from .abstract import Loss


class DynaQLearning(UniformModelPriorMixin, EnvFunctionsMixin, Loss):
    """Loss function Dyna-augmented Clipped Double Q-learning.

    Attributes:
        critics: Main action-values
        target_critic: Target state-value function
        models: Stochastic model ensemble
        batch_keys: Keys required to be in the tensor batch
        gamma: discount factor
    """

    critics: QValueEnsemble
    target_critic: VValue
    models: Union[StochasticModel, SME]

    gamma: float = 0.99
    batch_keys: Tuple[str] = (SampleBatch.CUR_OBS, SampleBatch.ACTIONS)
    _model_samples: int = 1

    def __init__(
        self,
        critics: QValueEnsemble,
        target_critic: VValue,
        models: Union[StochasticModel, SME],
    ):
        super().__init__()
        self.critics = critics
        self.target_critic = target_critic

        if isinstance(models, StochasticModel):
            # Treat everything as if ensemble
            models = SME([models])
        self.models = models

        self._loss_fn = nn.MSELoss()

    @property
    def model_samples(self) -> int:
        """Number of next states to sample from model."""
        return self._model_samples

    @model_samples.setter
    def model_samples(self, value: int):
        assert value > 0, "Number of model samples must be positive"
        self._model_samples = value

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        assert self._env.initialized, (
            "Environment functions missing. "
            "Did you set reward and termination functions?"
        )
        obs, action = self.unpack_batch(batch)

        with torch.no_grad():
            model, _ = self.sample_model()
            dist_params = model(obs, action)
            next_obs, _ = model.sample(dist_params, sample_shape=(self.model_samples,))

            reward = self._env.reward(obs, action, next_obs).mean(dim=0)
            next_values = self.target_critic(next_obs)
            dones = self._env.termination(obs, action, next_obs)
            next_values = torch.where(dones, torch.zeros_like(next_values), next_values)

            target = reward + self.gamma * next_values.mean(dim=0)

        values = self.critics(obs, action)
        loss = torch.stack([self._loss_fn(target, v) for v in values]).sum()

        stats = {"loss(critics)": loss.item()}
        stats.update(QLearningMixin.q_value_info(values))
        stats.update(dist_params_stats(dist_params, name="model"))
        return loss, stats
