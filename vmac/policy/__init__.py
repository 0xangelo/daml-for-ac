# pylint:disable=missing-docstring
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
from raylab.policy.modules.model.stochastic.single import StochasticModel
from raylab.utils.types import TensorDict
from torch import Tensor
from torch.jit import fork
from torch.jit import wait

SampleLogp = Tuple[Tensor, Tensor]


class CustomSME(nn.ModuleList):
    # pylint:disable=abstract-method,arguments-differ
    def __init__(self, models: List[StochasticModel]):
        cls_name = type(self).__name__
        assert all(
            isinstance(m, StochasticModel) for m in models
        ), f"All modules in {cls_name} must be instances of StochasticModel."
        super().__init__(models)

    def forward(self, obs: List[Tensor], act: List[Tensor]) -> List[TensorDict]:
        futures = [fork(m, obs[i], act[i]) for i, m in enumerate(self)]
        return [wait(f) for f in futures]

    @torch.jit.export
    def sample(self, obs: List[Tensor], act: List[Tensor]) -> List[SampleLogp]:
        futures = [fork(m.sample, obs[i], act[i]) for i, m in enumerate(self)]
        return [wait(f) for f in futures]

    @torch.jit.export
    def rsample(self, obs: List[Tensor], act: List[Tensor]) -> List[SampleLogp]:
        futures = [fork(m.rsample, obs[i], act[i]) for i, m in enumerate(self)]
        return [wait(f) for f in futures]

    @torch.jit.export
    def log_prob(
        self, obs: List[Tensor], act: List[Tensor], new_obs: List[Tensor]
    ) -> List[Tensor]:
        futures = [
            fork(m.log_prob, obs[i], act[i], new_obs[i]) for i, m in enumerate(self)
        ]
        return [wait(f) for f in futures]

    @torch.jit.export
    def sample_from_params(self, dist_params: List[TensorDict]) -> List[SampleLogp]:
        futures = [fork(m.dist.sample, dist_params[i]) for i, m in enumerate(self)]
        return [wait(f) for f in futures]

    @torch.jit.export
    def rsample_from_params(self, dist_params: List[TensorDict]) -> List[SampleLogp]:
        futures = [fork(m.dist.rsample, dist_params[i]) for i, m in enumerate(self)]
        return [wait(f) for f in futures]

    @torch.jit.export
    def log_prob_from_params(
        self, obs: List[Tensor], params: List[TensorDict]
    ) -> List[Tensor]:
        futures = [fork(m.dist.log_prob, obs[i], params[i]) for i, m in enumerate(self)]
        return [wait(f) for f in futures]
