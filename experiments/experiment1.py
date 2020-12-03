# pylint:disable=missing-docstring,import-outside-toplevel
import itertools
import os
import typing as ta

import click
import pytorch_lightning as pl
import ray
import raylab
import wandb  # pylint:disable=wrong-import-order
from pytorch_lightning.loggers import WandbLogger
from raylab.policy import TorchPolicy
from raylab.policy.model_based.lightning import ReplayDataset
from raylab.utils.replay_buffer import NumpyReplayBuffer
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import vmac

import models as mods  # noreorder
import utils  # noreorder


# ================================================================================
# DATA MODULE
# ================================================================================


class DataModule(pl.LightningDataModule):
    def __init__(self, run: utils.Run, replay: NumpyReplayBuffer, **loader_kwargs):
        super().__init__()
        self.data = ReplayDataset(replay)
        run.config.timesteps = len(self.data)
        self.loader_kwargs = loader_kwargs

        self.split = (0.7, 0.2, 0.1)

    def setup(self, _):
        size = len(self.data)
        split_sizes = list(map(round, (s * size for s in self.split)))
        split_sizes[-1] = size - sum(split_sizes[:2])
        self.train_data, self.val_data, self.test_data = random_split(
            self.data, split_sizes
        )

    def train_dataloader(self):
        kwargs = {**self.loader_kwargs, "shuffle": True}
        return DataLoader(self.train_data, **kwargs)

    def val_dataloader(self):
        kwargs = {**self.loader_kwargs, "shuffle": False}
        return DataLoader(self.val_data, **kwargs)

    def test_dataloader(self):
        kwargs = {**self.loader_kwargs, "shuffle": False}
        return DataLoader(self.test_data, **kwargs)


# ================================================================================
# VALUE-AWARE MODEL FROM MODEL-FREE MODULES
# ================================================================================


class LightningModel(mods.LightningModel):
    def __init__(self, run: utils.Run, policy: TorchPolicy):
        run.config.env = policy.config["env"]

        model = mods.build_single(
            policy.observation_space, policy.action_space, self.get_model_spec(run)
        )

        value = utils.get_value_from_policy(policy)
        assert not run.config.std_value
        loss = mods.VAML(model, value)
        loss.grad_estimator = run.config.grad_estimator.upper()
        loss.model_samples = run.config.model_samples
        loss.alpha = run.config.alpha
        if run.config.script:
            loss.compile()

        super().__init__(model, loss)
        self.hparams.learning_rate = run.config.lr

    @classmethod
    def get_model_spec(
        cls, run: utils.Run
    ) -> ta.Union[mods.StandardSpec, mods.FixedSpec]:
        if run.config.fixed_scale:
            spec = mods.FixedSpec()
            spec.network.scale = 0.1
        else:
            spec = mods.StandardSpec()
            spec.network.fix_logvar_bounds = True
            spec.network.input_dependent_scale = True

        spec.network.units = (32, 32)
        spec.network.activation = "Swish"
        spec.network.delay_action = False
        spec.residual = True
        spec.initializer = {"name": "orthogonal"}
        return spec


# ================================================================================
# COMMAND LINE INTERFACE
# ================================================================================


def get_artifact(run: utils.Run, artifact_name: str) -> ta.Any:
    artifact = run.use_artifact(artifact_name)
    artifact_dir = artifact.download()

    print("Artifact id:", artifact.id)
    print("Artifact name:", artifact.name)
    print("Artifact dirname:", artifact_dir)
    print(os.listdir(artifact_dir))

    return artifact


def get_policy(artifact: ta.Any) -> TorchPolicy:
    policy = utils.get_policy_from_artifact(artifact)
    policy.replay = utils.single_precision_replay(policy.replay)
    utils.restore_replay_from_artifact(artifact, policy.replay)
    return policy


def run_model_training(run, artifact: str):
    art = get_artifact(run, artifact)
    policy = get_policy(art)

    model = LightningModel(run, policy)
    data = DataModule(run, policy.replay, batch_size=128)

    logger = WandbLogger(log_model=True, experiment=run)
    trainer = pl.Trainer(
        max_epochs=run.config.max_epochs,
        logger=logger,
        early_stop_callback=pl.callbacks.EarlyStopping(patience=run.config.patience),
        progress_bar_refresh_rate=0,
    )
    logger.watch(model, log="all")

    trainer.fit(model, datamodule=data)
    trainer.test(model=model, datamodule=data)


@ray.remote(num_cpus=1)
def experiment(artifact: str, **options):
    run = utils.wandb_run_initializer(
        project="vaml", entity="angelovtt", tags=["exp1", "new loss"], reinit=True
    )(**options)

    with run:
        run_model_training(run, artifact)


def checkpoint_iter(step: int = 1) -> ta.Iterable[str]:
    prefix = "angelovtt/baselines/"
    agent = "SoftAC"
    # envs = ["Hopper-v3", "Walker2d-v3"]
    envs = ["Hopper-v3"]
    iterations = [it * 10 for it in range(1, 20, step)]
    checkpoints = [
        f"{prefix}{agent}_{env}_{it}ts:latest"
        for env, it in itertools.product(envs, iterations)
    ]
    return checkpoints


@click.command()
@click.argument("artifacts", type=str, nargs=-1)
@click.option(
    "--checkpoint-step",
    type=int,
    default=1,
    show_default=True,
    help="Step size when iterating over checkpoints.",
)
@click.option(
    "--grad-estimator",
    type=click.Choice("PD SF".split(), case_sensitive=False),
    default="pd",
    show_default=True,
)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--patience", type=int, default=10, show_default=True)
@click.option("--model-samples", type=int, default=10, show_default=True)
@click.option("--std-value/--raw-value", default=False, show_default=True)
@click.option("--fixed-scale/--learn-scale", default=False, show_default=True)
@click.option("--alpha", "-a", type=float, multiple=True)
@click.option("--script/--eager", default=False, show_default=True)
@click.option("--max-epochs", type=int, default=1000, show_default=True)
def main(
    artifacts: ta.Tuple[str, ...],
    checkpoint_step: int,
    alpha: ta.Tuple[float, ...],
    **options,
):
    ray.init()
    raylab.register_all()
    vmac.register_all()

    futures = []
    checkpoints = artifacts or checkpoint_iter(step=checkpoint_step)
    for chkpt in checkpoints:
        for alp in alpha:
            futures += [experiment.remote(chkpt, alpha=alp, **options)]
    ray.get(futures)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
