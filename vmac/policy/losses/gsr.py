"""Losses for Gradient-Supervised Regression."""
from typing import Tuple

import torch
import torch.nn as nn
from ray.rllib import SampleBatch
from raylab.policy.losses import MaximumLikelihood
from raylab.policy.losses.mixins import EnvFunctionsMixin
from raylab.policy.modules.actor.policy.stochastic import StochasticPolicy
from raylab.policy.modules.critic.q_value import QValueEnsemble
from raylab.policy.modules.model.stochastic.ensemble import SME
from raylab.utils.types import StatDict
from raylab.utils.types import TensorDict
from torch import Tensor
from torch.autograd import grad

from .abstract import Loss


class GSR(EnvFunctionsMixin, Loss):
    """Gradient-Supervised Regression for model learning w/ maximum entropy.

    Computes the decision-aware loss for model ensembles used in Model-Aware
    Policy Optimization.

    Args:
        models: The stochastic model ensemble
        actor: The stochastic policy
        critics: The action-value estimators

    Attributes:
        gamma: Discount factor
        alpha: Entropy regularization coefficient
        grad_estimator: Gradient estimator for expecations ('PD' or 'SF')
        manhattan: Whether to compute the action gradient's 1-norm or
            squared error
        lambda_: Kullback Leibler regularization coefficient

    Note:
        `N` denotes the size of the model ensemble, `O` the size of the
        observation, and `A` the size of the action
    """

    batch_keys: Tuple[str, str, str] = (
        SampleBatch.CUR_OBS,
        SampleBatch.ACTIONS,
        SampleBatch.NEXT_OBS,
    )
    gamma: float = 0.99
    alpha: float = 0.05
    grad_estimator: str = "SF"
    manhattan: bool = False
    lambda_: float = 0.05
    _last_output: Tuple[Tensor, StatDict]

    def __init__(
        self,
        models: SME,
        actor: StochasticPolicy,
        critics: QValueEnsemble,
    ):
        super().__init__()
        modules = nn.ModuleDict()
        modules["models"] = models
        modules["policy"] = actor
        modules["critics"] = critics
        self._modules = modules
        self._loss_mle = MaximumLikelihood(models)

    @property
    def initialized(self) -> bool:
        """Whether or not the loss setup is complete."""
        return self._env.initialized

    @property
    def ensemble_size(self) -> int:
        """The number of models in the ensemble."""
        return len(self._modules["models"])

    @property
    def last_output(self) -> Tuple[Tensor, StatDict]:
        """Last computed losses for each individual model and associated info."""
        return self._last_output

    def compile(self):
        self._loss_mle.compile()

    def __call__(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        assert self.initialized, (
            "Environment functions missing. "
            "Did you set reward and termination functions?"
        )
        obs = batch[SampleBatch.CUR_OBS]
        obss = self.expand_foreach_model(obs)
        actions = self.generate_action(obss)
        value_targets = self.zero_step_action_value(obss, actions)
        value_preds = self.one_step_action_value_surrogate(obss, actions)
        grad_loss = self.action_gradient_loss(actions, value_targets, value_preds)
        mle_loss, info = self.maximum_likelihood_loss(batch)

        loss = grad_loss + self.lambda_ * mle_loss
        info.update({f"loss(models[{i}])": s for i, s in enumerate(loss.tolist())})
        info["loss(daml)"] = grad_loss.mean().item()
        info["loss(mle)"] = mle_loss.mean().item()
        self._last_output = (loss, info)
        return loss.mean(), info

    def expand_foreach_model(self, tensor: Tensor) -> Tensor:
        """Add first dimension to tensor with the size of the model ensemble.

        Args:
            tensor: Tensor of shape `S`

        Returns:
            Tensor `tensor` expanded to shape `(N,) + S`
        """
        return tensor.expand((self.ensemble_size,) + tensor.shape)

    @torch.no_grad()
    def generate_action(self, obs: Tensor) -> Tensor:
        """Given state, sample action with the stochastic policy.

        Generates one action for each of the `N` models in the ensemble, so that
        action gradients and subsequent losses may be calculated for each model.

        Args:
            obs: The current observation tensor of shape `(N, *, O)`

        Returns:
            The action tensor of shape `(N, *, A)` with `requires_grad` enabled
        """
        action, _ = self._modules["policy"].sample(obs)
        return action.requires_grad_(True)

    def zero_step_action_value(self, obs: Tensor, action: Tensor) -> Tensor:
        """Compute action-value directly using approximate critic.

        Calculates :math:`Q^{\\pi}(s, a)` as a target for each model in the
        ensemble. Each value is the minimum among critic predictions.

        Args:
            obs: The observation tensor of shape `(N, *, O)`
            action: The action Tensor of shape `(N, *, A)`

        Returns:
            The action-value tensor of shape `(N, *)`
        """
        return QValueEnsemble.clipped(self._modules["critics"](obs, action))

    def one_step_action_value_surrogate(self, obs: Tensor, action: Tensor) -> Tensor:
        """Surrogate loss for gradient estimation of action values.

        Args:
            obs: The observation tensor of shape `(N, *, O)`
            action: The action Tensor of shape `(N, *, A)`

        Returns:
            A tensor of shape `(N, *)` for estimating the gradient of the 1-step
            action-value function.
        """
        next_obs, next_obs_logp = self.transition(obs, action)
        next_act, next_act_logp = self._modules["policy"].rsample(next_obs)

        next_qval = QValueEnsemble.clipped(self._modules["critics"](next_obs, next_act))

        reward = self._env.reward(obs, action, next_obs)
        done = self._env.termination(obs, action, next_obs)

        next_vval = (
            torch.where(done, reward, reward + self.gamma * next_qval)
            - self.alpha * next_act_logp
        )

        if self.grad_estimator == "SF":
            # FIX: Wrong use of the score function estimator
            surrogate = next_obs_logp * next_vval.detach()
        elif self.grad_estimator == "PD":
            surrogate = next_vval
        return surrogate

    def transition(self, obs: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute virtual transition and its log density.

        Args:
            obs: The current state tensor of shape `(N, *, O)`
            action: The action tensor of shape `(N, *, A)` sampled from the
                stochastic policy

        Returns:
            A tuple with the next state tensor of shape `(N, *, O)` and its
            log-likelihood tensor of shape `(N, *)` generated from models in the
            ensemble
        """
        models = self._modules["models"]
        obs = obs.chunk(self.ensemble_size, dim=0)
        action = action.chunk(self.ensemble_size, dim=0)

        params = models(obs, action)
        if self.grad_estimator == "SF":
            outputs = models.sample(params)
        elif self.grad_estimator == "PD":
            outputs = models.rsample(params)

        next_obs, logp = zip(*outputs)
        return torch.cat(next_obs), torch.cat(logp)

    def action_gradient_loss(
        self, action: Tensor, value_target: Tensor, value_pred: Tensor
    ) -> Tensor:
        """Decision-aware model loss based on action gradients.

        Args:
            action: The action tensor of shape `(N, *, A)`
            value_target: The estimated action-value gradient target tensor of
                shape `(N, *)`
            value_pred: The surrogate loss tensor of shape `(N, *)` for action
                gradient estimation of the 1-step action-value

        Returns:
            The loss tensor of shape `(N,)`
        """
        temporal_diff = value_target - value_pred
        (action_gradient,) = grad(temporal_diff.sum(), action, create_graph=True)

        # First compute action gradient norms by reducing along action dimension
        if self.manhattan:
            grad_norms = action_gradient.abs().sum(dim=-1)
        else:
            grad_norms = torch.sum(action_gradient ** 2, dim=-1) / 2
        # Return mean action gradient loss along batch dimension
        return grad_norms.mean(dim=-1)

    def maximum_likelihood_loss(self, batch: TensorDict) -> Tuple[Tensor, StatDict]:
        """Model regularization through Maximum Likelihood.

        Args:
            batch: The tensor batch with SampleBatch.CUR_OBS,
                SampleBatch.ACTIONS, and SampleBatch.NEXT_OBS keys

        Returns:
            The loss tensor of shape `(N,)` and associated info dict
        """
        self._loss_mle(batch)
        return self._loss_mle.last_output
