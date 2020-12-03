# pylint:disable=missing-docstring
import math
import typing as ta
import warnings

import click
import environments
import gym
import models as mods
import numpy as np
import pytorch_lightning as pl
import ray
import raylab.envs
import torch
import utils
from ray.rllib import SampleBatch
from raylab.policy.losses.abstract import Loss
from raylab.policy.modules.actor.policy.deterministic import DeterministicPolicy
from raylab.torch.nn.utils import update_polyak
from raylab.torch.utils import TensorDictDataset
from raylab.utils.replay_buffer import NumpyReplayBuffer
from raylab.utils.types import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader

import vmac
from vmac.policy.losses import DynaCDQLearning


# ======================================================================================
# GROUND-TRUTH Q-VALUE
# ======================================================================================


# def action_value_error_fn(
#     run, policy: utils.SOPTorchPolicy, replay: NumpyReplayBuffer
# ) -> ta.Callable[[], dict]:
#     batch = test_data(run, policy, replay)
#     # Consider one QValue for now
#     critic = policy.module.critics[0]

#     @torch.no_grad()
#     def action_value_error():
#         targets = batch[TARGET_Q]
#         preds = critic(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])
#         assert preds.shape == targets.shape

#         errs = torch.abs(targets - preds).mean().item()
#         prefix = "test/"
#         results = {
#             "mean_abs_Q_err": errs,
#             "sim_returns": wandb.Histogram(targets.numpy()),
#             "pred_returns": wandb.Histogram(preds.numpy()),
#         }
#         return {prefix + k: v for k, v in results.items()}

#     return action_value_error


# ======================================================================================
# DATA
# ======================================================================================


TARGET_Q = "target_q"
SCALARS = {TARGET_Q, SampleBatch.REWARDS, SampleBatch.DONES}


def get_action_value_fn(
    env=gym.Env, gamma: float = 0.99, horizon=None
) -> ta.Callable[[DeterministicPolicy, Tensor], ta.Tuple[Tensor, Tensor, Tensor]]:
    horizon = horizon or round(-1 / math.log(gamma))
    assert horizon > 0

    @torch.no_grad()
    def action_values(policy: DeterministicPolicy, obs: Tensor) -> TensorDict:
        n_trajs = len(obs)
        trajs = {
            "cur_done": [],
            SampleBatch.CUR_OBS: [],
            SampleBatch.ACTIONS: [],
            SampleBatch.REWARDS: [],
            SampleBatch.DONES: [],
            SampleBatch.NEXT_OBS: [],
        }

        done = torch.zeros(obs.shape[:-1]).bool()
        for _ in range(horizon):
            act = policy(obs)
            new_obs, _ = env.transition_fn(obs, act)
            reward = env.reward_fn(obs, act, new_obs)
            reward = torch.where(done, torch.zeros_like(reward), reward)
            # Retain done if termination was hit earlier
            new_done = env.termination_fn(obs, act, new_obs) | done

            trajs["cur_done"] += [done]
            trajs[SampleBatch.CUR_OBS] += [obs]
            trajs[SampleBatch.ACTIONS] += [act]
            trajs[SampleBatch.REWARDS] += [reward]
            trajs[SampleBatch.DONES] += [new_done]
            trajs[SampleBatch.NEXT_OBS] += [new_obs]

            obs = new_obs
            done = new_done

        returns = torch.stack(trajs[SampleBatch.REWARDS], dim=0)  # (horizon, n_trajs)
        for time in reversed(range(horizon - 1)):
            returns[time] += gamma * returns[time + 1]

        trajs = {k: torch.stack(v, dim=0) for k, v in trajs.items()}
        trajs[TARGET_Q] = returns
        trajs = {k: v.reshape(horizon * n_trajs, -1) for k, v in trajs.items()}
        cur_done = trajs.pop("cur_done")
        trajs = {k: v[~cur_done.squeeze(-1)] for k, v in trajs.items()}
        trajs = {k: v.squeeze(-1) if k in SCALARS else v for k, v in trajs.items()}
        return trajs

    return action_values


def get_env_with_dynamics(env_id: str, env_config: dict) -> gym.Env:
    env = raylab.envs.get_env_creator(env_id)(env_config)
    if env_config.get("time_aware", False):
        env = environments.TimeAwareTransitionFn(env)
    reward_fn = raylab.envs.get_reward_fn(env_id, env_config)
    termination_fn = raylab.envs.get_termination_fn(env_id, env_config)

    setattr(env, "reward_fn", reward_fn)
    setattr(env, "termination_fn", termination_fn)
    return env


def test_data(run, policy: utils.SOPTorchPolicy) -> TensorDict:
    config = policy.config
    env = get_env_with_dynamics(env_id=config["env"], env_config=config["env_config"])
    action_value_fn = get_action_value_fn(
        env=env,
        gamma=config["gamma"],
        horizon=config["env_config"].get("max_episode_steps"),
    )

    # Always evaluate on the same set of initial states
    obs = policy.convert_to_tensor(
        np.stack([env.reset() for _ in range(run.config.eval_trajs)])
    )
    batch = action_value_fn(policy.module.actor, obs)
    assert len(batch[TARGET_Q].shape) == 1
    print("BATCH:", batch)
    return batch


class DataModule(utils.DataModule):
    # pylint:disable=abstract-method
    def __init__(self, run: utils.Run, agent, val_ratio: float = 0.2, **loader_kwargs):
        replay: NumpyReplayBuffer = agent.replay
        super().__init__(replay, val_ratio, **loader_kwargs)

        # Get data on init to avoid generating new trajectories on every call to setup
        # (when passed to Trainer)
        policy: utils.SOPTorchPolicy = agent.get_policy()
        self.test_data = test_data(run, policy)
        self.test_datasets = None

    def setup(self, _):
        super().setup(_)
        self.test_datasets = (TensorDictDataset(self.test_data), self.replay_dataset)

    def test_dataloader(self):  # pylint:disable=arguments-differ
        kwargs = {"shuffle": False, **self.loader_kwargs}
        traj_data, replay_data = self.test_datasets
        return [DataLoader(traj_data, **kwargs), DataLoader(replay_data, **kwargs)]


# ======================================================================================
# MODEL
# ======================================================================================


class LightningModel(mods.LightningModel):
    # pylint:disable=too-many-ancestors
    def __init__(self, run, policy: utils.SOPTorchPolicy):
        value = utils.get_value_from_policy(policy)
        model = mods.build_single(
            policy.observation_space, policy.action_space, self.get_model_spec(run)
        )

        assert not run.config.std_value
        loss = mods.VAML2(model, value)
        loss.grad_estimator = run.config.grad_estimator.upper()
        loss.model_samples = run.config.model_samples
        loss.alpha = run.config.alpha

        super().__init__(model, loss)

        self.hparams.learning_rate = run.config.lr
        self.policy = policy

    @staticmethod
    def get_model_spec(run: utils.Run) -> ta.Union[mods.StandardSpec, mods.FixedSpec]:
        if run.config.fixed_scale:
            spec = mods.FixedSpec()
            spec.network.scale = 0.1
        else:
            spec = mods.StandardSpec()
            spec.network.fix_logvar_bounds = True
            spec.network.input_dependent_scale = True

        spec.network.units = (128, 128)
        spec.network.activation = "Swish"
        spec.network.delay_action = False
        # MBPO fits scaler on entire training dataset, should we?
        spec.network.standard_scaler = False
        spec.residual = True
        spec.initializer = {"name": "orthogonal"}
        return spec

    def test_step(self, batch, *args):
        result = super().test_step(batch, *args)

        if TARGET_Q in batch:
            targets = batch[TARGET_Q]
            # Consider one QValue for now
            critic = self.policy.module.critics[0]
            preds = critic(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS])
            assert preds.shape == targets.shape
            errs = torch.abs(targets - preds).mean()

            result.log("test/mean_abs_Q_err", errs)
        return result


# ======================================================================================
# ALGORITHM
# ======================================================================================


def test_model(model: mods.LightningModel, data: DataModule) -> dict:
    info = {}
    trainer = pl.Trainer(max_epochs=0, progress_bar_refresh_rate=0, logger=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        (trajs, replay) = trainer.test(model, datamodule=data)
        assert isinstance(trajs, dict)
        assert isinstance(replay, dict)
        info.update({k.replace("test", "trajs"): v for k, v in trajs.items()})
        info.update({k.replace("test", "replay"): v for k, v in replay.items()})

    return info


def get_critic_loss(policy: utils.SOPTorchPolicy, model: mods.LightningModel) -> Loss:
    config = policy.config
    critic_loss = DynaCDQLearning(
        critics=policy.module.critics,
        actor=policy.module.actor,
        models=model.model,
        target_critic=utils.get_value_from_policy(policy),
    )
    critic_loss.set_reward_fn(
        raylab.envs.get_reward_fn(config["env"], config["env_config"])
    )
    critic_loss.set_termination_fn(
        raylab.envs.get_termination_fn(config["env"], config["env_config"])
    )
    return critic_loss


def policy_evaluation(
    policy: utils.SOPTorchPolicy, model: mods.LightningModel
) -> ta.Callable[[SampleBatch], dict]:
    loss_fn = get_critic_loss(policy, model)

    def policy_evaluator(samples: SampleBatch) -> dict:
        batch = policy.lazy_tensor_dict(samples)
        with policy.optimizers.optimize("critics"):
            loss, log = loss_fn(batch)
            loss.backward()

        update_polyak(
            from_module=policy.module.critics,
            to_module=policy.module.target_critics,
            polyak=0.995,
        )
        return log

    return policy_evaluator


def update_model(run, model: mods.LightningModel, data: utils.DataModule) -> dict:
    trainer = pl.Trainer(
        max_epochs=1000,
        progress_bar_refresh_rate=0,
        early_stop_callback=pl.callbacks.EarlyStopping(patience=run.config.patience),
        logger=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(model, datamodule=data)


def learn(run, agent):
    rollout_worker = agent.workers.local_worker()

    data = DataModule(run, agent, batch_size=128)
    model = LightningModel(run, agent.get_policy())

    update_model(run, model, data)
    log = dict(step=0)
    log.update(test_model(model, data))
    run.log(log)

    policy_evaluator = policy_evaluation(agent.get_policy(), model)
    for step in range(1, run.config.env_steps + 1):
        log = {"step": step}

        samples = rollout_worker.sample()
        assert samples.count == 1
        for row in samples.rows():
            agent.replay.add(row)
        assert len(data.replay_dataset) == len(agent.replay)

        if step % run.config.model_interval == 0:
            update_model(run, model, data)

        for _ in range(run.config.policy_evals_per_step):
            log.update(policy_evaluator(agent.replay.sample(128)))

        if step % run.config.eval_interval == 0:
            log.update(test_model(model, data))

        run.log(log)


# ======================================================================================
# EXPERIMENT OUTLINE
# ======================================================================================


@ray.remote(num_cpus=1)
def experiment(artifact_id: str, **options):
    run = utils.wandb_run_initializer(
        project="vaml", entity="angelovtt", tags=["exp3"], reinit=True
    )(artifact=artifact_id, **options)

    with run:
        artifact = run.use_artifact(artifact_id)
        agent = utils.get_agent_from_artifact(artifact, agent_name="SOP")
        utils.restore_replay_from_artifact(artifact, agent)
        run.config.init_timesteps = len(agent.replay)

        learn(run, agent)


# ======================================================================================
# CLICK INTERFACE
# ======================================================================================


@click.command()
@click.argument("artifacts", type=str, nargs=-1)
@click.option("--lr", type=float, default=1e-4, show_default=True)
@click.option("--patience", type=int, default=10, show_default=True)
@click.option("--std-value/--raw-value", default=False, show_default=True)
@click.option("--fixed-scale/--learn-scale", default=False, show_default=True)
@click.option(
    "--env-steps",
    type=int,
    default=1000,
    show_default=True,
    help="Total number of environment steps",
)
@click.option(
    "--policy-evals-per-step",
    type=int,
    default=10,
    show_default=True,
    help="Policy Evaluation steps per environment step",
)
@click.option(
    "--eval-interval",
    type=int,
    default=10,
    show_default=True,
    help="Environment steps between each model evaluation",
)
@click.option("--alpha", "-a", type=float, multiple=True)
@click.option(
    "--grad-estimator",
    type=click.Choice("SF PD".split(), case_sensitive=False),
    default="PD",
    show_default=True,
)
@click.option("--model-samples", type=int, default=1, show_default=True)
@click.option("--model-interval", type=int, default=25, show_default=True)
@click.option("--eval-trajs", type=int, default=10, show_default=True)
def main(artifacts: ta.Tuple[str, ...], alpha: ta.Tuple[float, ...], **options):
    """Launch Experiment 3.

    All options are forwarded to W&B's run config and are thus accessible via
    `run.config.<option>`
    """
    ray.init()
    raylab.register_all()
    vmac.register_all()

    futures = []
    for artifact in artifacts:
        for alp in alpha:
            futures += [experiment.remote(artifact, alpha=alp, **options)]
    ray.get(futures)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
