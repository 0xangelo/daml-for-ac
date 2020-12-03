# pylint:disable=missing-docstring
import logging
import typing as ta

import click
import models as mods
import pytorch_lightning as pl
import ray
import raylab
import torch
import torch.nn as nn
import utils
import wandb  # pylint:disable=wrong-import-order
import yaml
from ray.rllib import SampleBatch
from raylab.utils.types import StatDict
from raylab.utils.types import TensorDict
from torch import Tensor

import vmac

logger = logging.getLogger(__name__)


def learn_and_eval(run, agent, model: mods.LightningModel, trainer: pl.Trainer):
    test_loader = utils.make_dataloader(agent.replay.all_samples(), batch_size=128)

    policy = agent.get_policy()
    new_value = utils.get_value_from_policy(
        policy, deterministic=not run.config.stochastic_value
    )
    test_loss = TestLoss(model.model, model.train_loss.value, new_value)
    test_loss.grad_estimator = run.config.grad_estimator.upper()
    test_loss.model_samples = run.config.model_samples
    model.test_loss = test_loss

    def eval_model(step):
        [results] = trainer.test(model, test_loader)
        results.update(mf_step=step)
        results["value_param_diff"] = torch.norm(
            nn.utils.parameters_to_vector(model.test_loss.init_value.parameters())
            - nn.utils.parameters_to_vector(model.test_loss.test_value.parameters()),
            p=1,
        ).item()
        run.log(results)

    eval_model(0)
    for step in range(1, run.config.mf_steps + 1):
        batch = agent.replay.sample(128)
        if run.config.policy_evaluation:
            info = policy.learn_critic(batch)
        else:
            info = policy.learn_on_batch(batch)
        info["mf_step"] = step
        run.log(info)

        if step % run.config.eval_interval == 0:
            eval_model(step)


class TestLoss(mods.VAML):
    def __init__(
        self,
        model: mods.StochasticModel,
        init_value: mods.VValue,
        test_value: mods.VValue,
    ):
        iparams = list(init_value.parameters())
        tparams = list(test_value.parameters())
        assert set(iparams).isdisjoint(set(tparams))
        assert all([torch.allclose(i, t) for i, t in zip(iparams, tparams)])

        super().__init__(model, value=test_value)
        self._init_value = init_value
        self.alpha = 0.5  # Assert both VAE and NLL are computed

    @property
    def init_value(self) -> mods.VValue:
        return self._init_value

    @property
    def test_value(self) -> mods.VValue:
        return self.value

    def __call__(self, batch: TensorDict) -> ta.Tuple[Tensor, StatDict]:
        # Assert both VAE and NLL are computed
        with self.eval():
            loss, info = super().__call__(batch)

        obs = batch[SampleBatch.CUR_OBS]
        extra = {
            "l1_value_diff": torch.norm(
                self.init_value(obs) - self.test_value(obs), p=1
            )
            .mean()
            .item()
        }
        info.update(extra)
        return loss, info


def build_model(obs_space, action_space) -> mods.StochasticModel:
    spec = mods.StandardSpec()
    spec.network.units = (128, 128)
    spec.network.activation = "Swish"
    spec.network.delay_action = False
    spec.network.standard_scaler = False
    spec.network.fix_logvar_bounds = True
    spec.network.input_dependent_scale = True
    spec.residual = True
    return mods.build_standard(obs_space, action_space, spec)


def setup_model(run, artifact: wandb.Artifact) -> mods.LightningModel:
    policy = utils.get_policy_from_artifact(artifact)
    run.config.env = policy.config["env"]

    model = build_model(policy.observation_space, policy.action_space)
    value = utils.get_value_from_policy(
        policy, deterministic=not run.config.stochastic_value
    )
    loss = mods.VAML(model, value)
    loss.grad_estimator = run.config.grad_estimator.upper()
    loss.model_samples = run.config.model_samples
    if run.config.train_with.lower() == "vaml":
        loss.alpha = 0.0
    else:
        loss.alpha = 1.0

    pl_model = mods.LightningModel(model, loss)

    return pl_model


def train_model(
    run, artifact: wandb.Artifact
) -> ta.Tuple[mods.LightningModel, pl.Trainer]:
    model = setup_model(run, artifact)

    replay = utils.get_replay_from_artifact(artifact)
    run.config.samples = replay.count
    replay.shuffle()
    train_loader, val_loader = utils.train_val_loaders(replay, batch_size=128)

    trainer = pl.Trainer(
        max_epochs=1000,
        early_stop_callback=pl.callbacks.EarlyStopping(patience=10),
        progress_bar_refresh_rate=0,
    )
    trainer.fit(model, train_loader, val_loader)
    return model, trainer


@ray.remote(num_cpus=1)
def experiment(artifact_id: str, **options):
    run = utils.wandb_run_initializer(
        project="vaml", entity="angelovtt", tags=["exp4"], reinit=True
    )(**options)

    with run:
        artifact = run.use_artifact(artifact_id)
        model, trainer = train_model(run, artifact)

        # Reload agent from artifact to get different parameters
        agent = utils.get_agent_from_artifact(artifact)
        utils.restore_replay_from_artifact(artifact, agent)

        learn_and_eval(run, agent, model, trainer)


# ======================================================================================
# CLICK INTERFACE
# ======================================================================================


@click.command()
@click.argument("artifact", type=str, nargs=-1)
@click.option(
    "--stochastic-value/--deterministic-value",
    default=True,
    show_default=True,
    help="Whether to use stochastic policy samples in estimating the state-value"
    " function",
)
@click.option(
    "--mf-steps",
    type=int,
    default=1000,
    show_default=True,
    help="""Total number of model-free steps.

    Each step can either be a Policy Evaluation or Policy Iteration step.
    See --policy-evaluation/--policy-iteration for more info.""",
)
@click.option(
    "--policy-evaluation/--policy-iteration",
    default=True,
    show_default=True,
    help="Whether perform Policy Evaluation or Policy Iteration in each "
    "model-free step",
)
@click.option(
    "--eval-interval",
    type=int,
    default=10,
    show_default=True,
    help="Policy Evaluation steps between each model evaluation",
)
@click.option(
    "--grad-estimator",
    type=click.Choice("SF PD".split(), case_sensitive=False),
    default="PD",
    show_default=True,
)
@click.option(
    "--model-samples",
    type=int,
    default=1,
)
def main(artifact: ta.Tuple[str, ...], **options):
    """Launch Experiment 4.2.

    All options are forwarded to W&B's run config and are thus accessible via
    `run.config.<option>`
    """
    ray.init()
    raylab.register_all()
    vmac.register_all()

    print("Options:")
    print(yaml.dump(options))

    futures = []
    for art in artifact:
        for train_with in "VAML MLE".split():
            # experiment(art, train_with=train_with, **options)
            futures += [experiment.remote(art, train_with=train_with, **options)]
    ray.get(futures)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
