# pylint:disable=missing-docstring
from typing import List
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from raylab.policy.losses.mle import LogVarReg
from raylab.policy.modules.critic.v_value import VValue
from raylab.policy.modules.model.stochastic.ensemble import ForkedSME
from raylab.policy.modules.model.stochastic.ensemble import SME
from raylab.policy.modules.model.stochastic.single import StochasticModel
from raylab.utils.types import StatDict
from raylab.utils.types import TensorDict
from torch import Tensor
from torch.jit import fork
from torch.jit import wait

from .abstract import Loss


# ======================================================================================
# Modules
# ======================================================================================


class PointWiseVAML(nn.Module):
    # pylint:disable=abstract-method
    def __init__(self, value: VValue):
        super().__init__()
        self.value = value

    def forward(self, obs: Tensor, pred_obs: Tensor) -> Tensor:
        target_value = self.value(obs)
        pred_values = self.value(pred_obs)
        # Average over next-state samples
        return 0.5 * (target_value - pred_values.mean(dim=0)) ** 2


class VAEstimator(nn.Module):
    # pylint:disable=abstract-method
    def __init__(self, model: StochasticModel, value: VValue, samples: int):
        super().__init__()
        self.model = model
        self.pointwise_costs = PointWiseVAML(value)
        self.samples = samples
        assert self.samples > 0

    def forward(self, params: TensorDict, new_obs: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class VAEScoreFunction(VAEstimator):
    # pylint:disable=abstract-method
    def forward(self, params: TensorDict, new_obs: Tensor) -> Tuple[Tensor, Tensor]:
        pred_obs, logps = self.model.sample(params, sample_shape=(self.samples,))
        costs = self.pointwise_costs(new_obs, pred_obs)

        # Weight each cost by the "upstream" grad log-probs
        # See Theorem 1 of "Gradient Estimation Using Stochastic Computation Graphs"
        logp = logps.sum(dim=0)
        surr, loss = torch.mean(costs.detach() * logp), costs.mean()
        return surr, loss


class VAEPathwiseDerivative(VAEstimator):
    # pylint:disable=abstract-method
    def forward(self, params: TensorDict, new_obs: Tensor) -> Tuple[Tensor, Tensor]:
        pred_obs, _ = self.model.rsample(params, sample_shape=(self.samples,))
        costs = self.pointwise_costs(new_obs, pred_obs)
        surr = loss = costs.mean()
        return surr, loss


class NLLLoss(nn.Module):
    """Compute Negative Log-Likelihood loss."""

    # pylint:disable=abstract-method

    def __init__(self, model: StochasticModel):
        super().__init__()
        self.model = model
        self.logvar_reg = LogVarReg()

    def forward(self, params: TensorDict, new_obs: Tensor) -> Tensor:
        # pylint:disable=missing-function-docstring
        logp = self.model.log_prob(new_obs, params)
        nll = -logp.mean()
        regularizer = self.logvar_reg(params)
        return nll + regularizer


class VALoss(nn.Module):
    # pylint:disable=abstract-method
    def __init__(self, model: StochasticModel, estimator: VAEstimator):
        super().__init__()
        self.model = model
        self.vae_estimator = estimator
        self.nll = NLLLoss(model)

    def forward(
        self, obs: Tensor, action: Tensor, new_obs: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, TensorDict]:
        params = self.model(obs, action)
        surr_vae = fork(self.vae_estimator, params, new_obs)
        nll = fork(self.nll, params, new_obs)
        surr, vae = wait(surr_vae)
        return surr, vae, wait(nll), params


class Losses(nn.ModuleList):
    # pylint:disable=abstract-method
    def __init__(self, losses: List[VALoss]):
        assert all(isinstance(loss, VALoss) for loss in losses)
        super().__init__(losses)

    def forward(
        self, obs: Tensor, action: Tensor, new_obs: Tensor
    ) -> List[Tuple[Tensor, Tensor, Tensor, TensorDict]]:
        # pylint:disable=arguments-differ
        return [loss(obs, action, new_obs) for loss in self]


class ForkedLosses(Losses):
    # pylint:disable=abstract-method
    def forward(
        self, obs: Tensor, action: Tensor, new_obs: Tensor
    ) -> List[Tuple[Tensor, Tensor, Tensor, TensorDict]]:
        futures = [fork(loss, obs, action, new_obs) for loss in self]
        return [wait(f) for f in futures]


# ======================================================================================
# Loss
# ======================================================================================


class VAML(Loss):
    # pylint:disable=too-many-instance-attributes
    batch_keys: Tuple[str, str, str] = (
        SampleBatch.CUR_OBS,
        SampleBatch.ACTIONS,
        SampleBatch.NEXT_OBS,
    )
    _grad_estimator: str = "PD"
    _model_samples: int = 1
    _alpha: float = 0.0
    _compiled: bool = False
    _last_output: Tuple[Tensor, StatDict]

    def __init__(self, models: Union[SME, StochasticModel], value: VValue):
        if isinstance(models, StochasticModel):
            # Treat everything as ensemble
            models = SME([models])

        self.models = models
        self.value = value
        self.build_losses()

    def build_losses(self):
        models = self.models
        cls = VAEPathwiseDerivative if self.grad_estimator == "PD" else VAEScoreFunction
        estimators = [cls(m, self.value, samples=self.model_samples) for m in models]
        losses = [VALoss(m, e) for m, e in zip(models, estimators)]
        self.losses = (
            ForkedLosses(losses) if isinstance(models, ForkedSME) else Losses(losses)
        )
        self.check_compiled()

    def check_compiled(self):
        # pylint:disable=attribute-defined-outside-init
        if self._compiled and not isinstance(self.losses, torch.jit.ScriptModule):
            self.losses = torch.jit.script(self.losses)

    def compile(self):
        self._compiled = True
        self.check_compiled()

    @property
    def grad_estimator(self) -> str:
        return self._grad_estimator

    @grad_estimator.setter
    def grad_estimator(self, value: str):
        valid = {"PD", "SF"}
        assert value in valid, f"Gradient estimator must be one of {valid}."
        old = self._grad_estimator
        self._grad_estimator = value
        if value != old:
            self.build_losses()

    @property
    def model_samples(self) -> int:
        return self._model_samples

    @model_samples.setter
    def model_samples(self, value: int):
        assert value > 0, "Number of model samples must be positive"
        old = self._model_samples
        self._model_samples = value
        if value != old:
            self.build_losses()

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        assert 0 <= value <= 1, "Alpha must be in [0, 1]"
        self._alpha = value

    @property
    def last_output(self) -> Tuple[Tensor, StatDict]:
        """Last computed losses for each individual model and associated info."""
        return self._last_output

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        info = {}
        vae_surrs, vaes, nlls, params_list = self.losses_from_batch(batch)

        surrs = torch.zeros(len(self.models))
        losses = torch.zeros(len(self.models))
        vae_weight, nll_weight = (1 - self.alpha), self.alpha
        if vae_weight > 0:
            surrs = surrs + vae_weight * vae_surrs
            losses = losses + vae_weight * vaes
        if nll_weight > 0:
            surrs = surrs + nll_weight * nlls
            losses = losses + nll_weight * nlls

        info.update(
            loss=losses.mean().item(),
            vae=vaes.mean().item(),
            nll=nlls.mean().item(),
            nll_weight=nll_weight,
        )
        info.update(self.dist_info(params_list))
        self._last_output = (losses, info)
        return surrs.mean(), info

    def losses_from_batch(
        self, batch: TensorDict
    ) -> Tuple[Tensor, Tensor, Tensor, List[TensorDict]]:
        obs, action, new_obs = self.unpack_batch(batch)
        outputs = self.losses(obs, action, new_obs)
        vae_surrs, vaes, nlls, params_list = zip(*outputs)
        vae_surrs, vaes, nlls = map(torch.stack, (vae_surrs, vaes, nlls))
        return vae_surrs, vaes, nlls, params_list

    @staticmethod
    @torch.no_grad()
    def dist_info(params: List[TensorDict]) -> StatDict:
        info = {}
        for idx, dist in enumerate(params):
            prefix = f"models[{idx}]/"
            if "loc" in dist:
                info[prefix + "loc-norm"] = dist["loc"].norm(p=1, dim=-1).mean()
            if "scale" in dist:
                info[prefix + "scale-norm"] = dist["scale"].norm(p=1, dim=-1).mean()
        return {k: v.item() for k, v in info.items()}
