"""
========================================================================================
SINGLE MODEL
========================================================================================
"""
# pylint:disable=missing-docstring
import typing as ta
from dataclasses import dataclass
from dataclasses import field

import raylab.torch.nn.distributions as ptd
import torch
import torch.nn as nn
from dataclasses_json import DataClassJsonMixin
from gym.spaces import Box
from raylab.policy.modules.model.stochastic.single import DynamicsParams
from raylab.policy.modules.model.stochastic.single import MLPModel
from raylab.policy.modules.model.stochastic.single import ResidualStochasticModel
from raylab.policy.modules.model.stochastic.single import StochasticModel
from raylab.policy.modules.networks.mlp import StateActionMLP


class NormalFixedScaleParams(nn.Module):
    """Produce Normal parameters with fixed scale."""

    # pylint:disable=abstract-method

    def __init__(self, in_features: int, event_size: int, scale: float = 1.0):
        super().__init__()
        self.event_shape = (event_size,)
        self.loc_module = nn.Linear(in_features, event_size)
        self.scale = scale

    def forward(self, inputs: torch.Tensor) -> ta.Dict[str, torch.Tensor]:
        # pylint:disable=arguments-differ
        loc = self.loc_module(inputs)
        shape = inputs.shape[:-1] + self.event_shape
        return {"loc": loc, "scale": torch.full(shape, self.scale)}


@dataclass
class SingleModelSpec(StateActionMLP.spec_cls):
    scale: float = 1.0


class FixedScaleModel(StochasticModel):
    # pylint:disable=abstract-method
    spec_cls = SingleModelSpec

    def __init__(self, obs_space: Box, action_space: Box, spec: SingleModelSpec):
        encoder = StateActionMLP(obs_space, action_space, spec)

        params = NormalFixedScaleParams(
            encoder.out_features, obs_space.shape[0], scale=spec.scale
        )
        params = DynamicsParams(encoder, params)
        dist = ptd.Independent(ptd.Normal(), reinterpreted_batch_ndims=1)

        super().__init__(params, dist)
        self.encoder = encoder

    def initialize_parameters(self, initializer_spec: dict):
        MLPModel.initialize_parameters(self, initializer_spec)


@dataclass
class Spec(DataClassJsonMixin):
    network: SingleModelSpec = field(default_factory=SingleModelSpec)
    residual: bool = True
    initializer: dict = field(default_factory=dict)


def build_single(obs_space: Box, action_space: Box, spec: Spec):
    model = FixedScaleModel(obs_space, action_space, spec.network)
    model.initialize_parameters(spec.initializer)
    return ResidualStochasticModel(model) if spec.residual else model
