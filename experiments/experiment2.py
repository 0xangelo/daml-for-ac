# pylint:disable=missing-docstring
import copy
import logging
import numbers
import os
import typing as ta

import click
import gym
import gym.spaces as spaces
import numpy as np
import pytorch_lightning as pl
import ray
import torch
import torch.nn as nn
from ray import tune
from ray.rllib import Policy
from ray.rllib import RolloutWorker
from ray.rllib import SampleBatch
from raylab.policy.losses import Loss
from raylab.policy.model_based.lightning import DataModule as BaseDatamod
from raylab.policy.model_based.lightning import DatamoduleSpec
from raylab.policy.model_based.lightning import ReplayDataset
from raylab.policy.modules.critic import MLPVValue
from raylab.policy.modules.model import StochasticModel
from raylab.torch.nn.init import initialize_
from raylab.utils.replay_buffer import NumpyReplayBuffer
from raylab.utils.types import StatDict
from raylab.utils.types import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader

from vmac.policy.losses import VAML

import models as mods  # noreorder
import utils  # noreorder


logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler())


def float_arr(size: int, fill_value: float) -> np.ndarray:
    return np.full(size, fill_value=fill_value, dtype=np.float32)


class FakeEnv(gym.Env):
    horizon = 200

    def __init__(self, run, dynamics: StochasticModel):
        self.observation_space, self.action_space = self.obs_act_spaces()
        self.dynamics = dynamics
        self._state: Tensor = torch.empty(self.observation_space.shape)
        self._timestep: int = 0
        self._deterministic: bool = run.config.deterministic_env

    @property
    def done(self) -> bool:
        return self._timestep >= self.horizon if self._timestep else False

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @staticmethod
    def obs_act_spaces() -> ta.Tuple[spaces.Box, spaces.Box]:
        # Imitate Hopper-v3 spaces
        obs_space = spaces.Box(low=float_arr(11, -np.inf), high=float_arr(11, np.inf))
        act_space = spaces.Box(low=float_arr(3, -1), high=float_arr(3, 1))
        # obs_space = spaces.Box(low=float_arr(2, -np.inf), high=float_arr(2, np.inf))
        # act_space = spaces.Box(low=float_arr(1, -1), high=float_arr(1, 1))
        return obs_space, act_space

    def reset(self) -> np.ndarray:
        self._state = torch.from_numpy(self.observation_space.sample())
        self._timestep = 0
        return self._get_obs()

    @torch.no_grad()
    def step(self, action: np.ndarray) -> ta.Tuple[np.ndarray, float, bool, dict]:
        state = self._state
        act = torch.from_numpy(action.astype(self.action_space.dtype, copy=False))

        dynamics = self.dynamics
        params = dynamics(state, act)
        transition = dynamics.deterministic if self.deterministic else dynamics.sample
        new_state, _ = transition(params)

        self._state = new_state
        self._timestep += 1

        return self._get_obs(), 0, self.done, {}

    def render(self, mode="human"):
        pass

    def _get_obs(self) -> np.ndarray:
        return self._state.numpy()


# ======================================================================================


class RandomUniformPolicy(Policy):
    # pylint:disable=abstract-method,too-many-arguments
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        **kwargs,
    ):
        return [self.action_space.sample() for _ in obs_batch], [], {}


def collect_samples(env: gym.Env, num: int) -> SampleBatch:
    worker = RolloutWorker(
        lambda _: env, RandomUniformPolicy, rollout_fragment_length=num
    )
    return worker.sample()


# ======================================================================================


class DataModule(BaseDatamod):
    # pylint:disable=abstract-method,too-many-instance-attributes

    def __init__(self, run, dynamics):
        logger.info("Collecting samples...")
        # Collect once since the environment changes between calls
        env = FakeEnv(run, dynamics)
        self.build_replay(env)

        spec = self.default_spec()
        super().__init__(self.train_replay, spec)

        self.full_dataset = self.replay_dataset
        self.test_dataset = ReplayDataset(self.test_replay)

        self.rng = np.random.default_rng()

    def build_replay(self, env):
        self.train_size, self.test_size = map(int, (5e4, 1e3))

        self.train_replay, self.test_replay = map(
            lambda s: NumpyReplayBuffer(
                env.observation_space, env.action_space, size=s
            ),
            (self.train_size, self.test_size),
        )
        train_samples, test_samples = self.train_test_samples(
            env, train_size=self.train_size, test_size=self.test_size
        )
        self.train_replay.add(train_samples)
        self.test_replay.add(test_samples)

    @staticmethod
    def train_test_samples(
        env: gym.Env, train_size: numbers.Number = 1e4, test_size: numbers.Number = 1e3
    ) -> ta.Tuple[SampleBatch, SampleBatch]:
        train_size, test_size = map(int, (train_size, test_size))
        samples = collect_samples(env, train_size + test_size)
        samples.shuffle()
        train_samples = samples.slice(0, train_size)
        test_samples = samples.slice(train_size, None)
        return train_samples, test_samples

    @staticmethod
    def default_spec() -> DatamoduleSpec:
        spec = DatamoduleSpec()
        spec.holdout_ratio = 0.2
        spec.max_holdout = None
        spec.batch_size = 128
        spec.shuffle = True
        spec.num_workers = 0
        return spec

    def subsample(self, size: int):
        indices = self.rng.permutation(self.train_size)[:size]
        self.replay_dataset = torch.utils.data.Subset(self.full_dataset, indices)

    def test_dataloader(self, *args, **kwargs):
        spec = self.spec
        kwargs = dict(
            shuffle=False, batch_size=spec.batch_size, num_workers=spec.num_workers
        )
        return DataLoader(self.test_dataset, **kwargs)


# ======================================================================================


class TestLoss(Loss):
    # pylint:disable=too-few-public-methods
    batch_keys: ta.Tuple[str, ...] = VAML.batch_keys

    def __init__(
        self,
        model: nn.Module,
        dynamics: nn.Module,
        loss: Loss,
    ):
        self.model = model
        self.dynamics = dynamics

        self._loss = loss

    def dist_diff(self, batch: TensorDict) -> StatDict:
        obs, act = (batch[k] for k in (SampleBatch.CUR_OBS, SampleBatch.ACTIONS))

        model_params = self.model(obs, act)
        true_params = self.dynamics(obs, act)
        model_loc, model_scale = model_params["loc"], model_params["scale"]
        true_loc, true_scale = true_params["loc"], true_params["scale"]

        loc_diff = torch.norm(true_loc - model_loc, p=1, dim=-1).mean()
        scale_diff = torch.norm(true_scale - model_scale, p=1, dim=-1).mean()
        scale_norm = torch.norm(model_scale, p=1, dim=-1).mean()
        info = {
            "loc-diff": loc_diff,
            "scale-diff": scale_diff,
            "scale-norm": scale_norm,
        }
        return {k: v.item() for k, v in info.items()}

    def __call__(self, batch: TensorDict) -> ta.Tuple[Tensor, StatDict]:
        loss, info = self._loss(batch)
        info.update(self.dist_diff(batch))
        return loss, info


class LightningModel(mods.LightningModel):
    model: StochasticModel
    env_dynamics: StochasticModel
    train_loss: Loss
    val_loss: Loss
    test_loss: Loss

    def __init__(self, run):
        model, dynamics = self.get_stochastic_models(run)
        value = self.value_fn(run)
        loss = self.make_vaml_loss(run, models=model, value=value)

        super().__init__(model, loss)

        self.test_loss = TestLoss(model, dynamics, loss)
        self.env_dynamics = dynamics
        self.hparams.realizable = run.config.realizable

    def get_stochastic_models(self, run) -> ta.Tuple[StochasticModel, StochasticModel]:
        obs_space, act_space = FakeEnv.obs_act_spaces()
        spec = self.architecture_spec(run)
        model_spec = spec.from_dict(spec.to_dict())
        model_spec.network.units = (
            spec.network.units if run.config.realizable else spec.network.units[:-1]
        )
        model = mods.build_single(obs_space, act_space, model_spec)
        dynamics = mods.build_single(obs_space, act_space, spec)
        return model, dynamics

    @staticmethod
    def architecture_spec(run) -> mods.StandardSpec:
        spec = mods.StandardSpec()
        spec.network.units = (run.config.hidden_size,) * 2
        spec.network.activation = None if run.config.linear else "Swish"
        spec.network.delay_action = False
        spec.network.fix_logvar_bounds = True
        spec.network.input_dependent_scale = True
        spec.residual = True
        return spec

    @staticmethod
    def value_fn(run) -> MLPVValue:
        obs_space, _ = FakeEnv.obs_act_spaces()
        spec = MLPVValue.spec_cls()
        spec.units = (64,)
        spec.activation = "ReLU"
        spec.layer_norm = False
        value = MLPVValue(obs_space, spec)
        # Large weights to force dissimilarities
        value.apply(
            initialize_(
                "orthogonal",
                activation=spec.activation,
                gain=run.config.value_init_gain,
            )
        )
        return value

    @staticmethod
    def make_vaml_loss(run, *args, **kwargs) -> VAML:
        loss = VAML(*args, **kwargs)
        loss.grad_estimator = run.config.grad_estimator
        loss.model_samples = run.config.model_samples
        loss.alpha = run.config.alpha
        return loss

    def param_diff(self) -> dict:
        if not self.hparams.realizable:
            return {"param_diff": np.nan}

        dynamics = self.env_dynamics
        model = self.model
        diff = torch.norm(
            nn.utils.parameters_to_vector(dynamics.parameters())
            - nn.utils.parameters_to_vector(model.parameters()),
            p=2,
        ).item()
        return {"param_diff": diff}


# ======================================================================================


def train_and_test(pl_model: LightningModel, datamodule: DataModule) -> dict:
    logger.info("Create Trainer with number of epochs")
    trainer = pl.Trainer(
        logger=False,
        progress_bar_refresh_rate=0,
        max_epochs=1000,
        early_stop_callback=True,
    )

    logger.info("Train models")
    trainer.fit(pl_model, datamodule=datamodule)

    logger.info("Test models")
    (test_result,) = trainer.test(pl_model, datamodule=datamodule)

    return test_result


def experiment(run):
    pl_model = LightningModel(run)
    # Same initial parameters for both models
    init_state = copy.deepcopy(pl_model.state_dict())

    datamodule = DataModule(run, pl_model.env_dynamics)

    for size in np.linspace(
        datamodule.train_size // 10, datamodule.train_size, num=11, dtype=np.int
    ):
        logger.info("Subsampling...")
        datamodule.subsample(size)

        logger.info("Reset models")
        pl_model.load_state_dict(init_state)

        results = train_and_test(pl_model, datamodule)
        results.update(pl_model.param_diff())
        results.update(samples=size)
        run.log(results)
        tune.report(size=size)


# ======================================================================================


def runner(options: dict):
    logger.setLevel(options["log_level"])
    init = utils.wandb_run_initializer(
        project="vaml", entity="angelovtt", tags=["exp2"], reinit=True
    )
    options.pop("log_level")
    run = init(**options)
    with run:
        experiment(run)


@click.command()
@click.option(
    "--log-level",
    type=click.Choice("DEBUG INFO WARN ERROR".split()),
    default="INFO",
    show_default=True,
)
def main(log_level: str):
    ray.init()

    config = {
        "log_level": log_level,
        "deterministic_env": False,
        "linear": False,
        "realizable": tune.grid_search([True, False]),
        "grad_estimator": "PD",
        "model_samples": 10,
        "value_init_gain": 1.0,
        "alpha": tune.grid_search([0.0, 1.0, 0.001]),
        "hidden_size": tune.grid_search([2 ** i for i in range(5, 11)]),
        "learn_scale": True,
    }
    tune.run(
        runner,
        name="Experiment2",
        config=config,
        local_dir=os.path.abspath("experiment2"),
        num_samples=4,
        loggers=[],
    )


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
