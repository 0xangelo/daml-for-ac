# pylint:disable=missing-docstring
import click
from common import tune_defaults
from common import wandb_config
from raylab.cli.utils import tune_options


def get_config(env: str, seed: int, warmup: int, evaluation_interval: int):
    config = {}
    config.update(env_config(env))
    config.update(trainer_config(seed, warmup, evaluation_interval))
    config.update(policy=policy_config(warmup))
    config.update(wandb=wandb_config(__file__))
    return config


def policy_config(warmup: int) -> dict:
    return {
        "module": module_config(),
        "optimizer": {
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},  # Taken from pfrl
        },
        "buffer_size": int(2e5),
        "batch_size": 256,
        "improvement_steps": 1,
        "polyak": 0.995,
        "target_entropy": "auto",
        "exploration_config": {"pure_exploration_steps": warmup},
    }


def module_config() -> dict:
    from raylab.policy.modules.sac import SACSpec

    spec = SACSpec()
    spec.actor.encoder.units = (256, 256)
    spec.actor.encoder.activation = "ReLU"
    spec.actor.encoder.layer_norm = False
    spec.actor.input_dependent_scale = True
    spec.actor.initial_entropy_coeff = 1.0  # Taken from pfrl
    spec.actor.initializer = {"name": "xavier_uniform"}

    spec.critic.encoder.units = (256, 256)
    spec.critic.encoder.activation = "ReLU"
    spec.critic.encoder.layer_norm = False
    spec.critic.double_q = True
    spec.critic.parallelize = True
    spec.critic.initializer = {"name": "xavier_uniform"}

    return {"type": "SAC", **spec.to_dict()}


def trainer_config(seed: int, warmup: int, evaluation_interval: int = 10) -> dict:
    return {
        "seed": seed,
        "gamma": 0.99,
        "learning_starts": warmup,
        "timesteps_per_iteration": 1000,
        "rollout_fragment_length": 1,
        "evaluation_interval": evaluation_interval,
        "evaluation_num_episodes": 10,
        "evaluation_config": {"explore": False},
    }


def env_config(env: str):
    conf = {
        "env": env,
        "env_config": {"time_aware": False, "max_episode_steps": 1000},
    }
    if env.endswith("-v3"):
        conf["env_config"]["exclude_current_positions_from_observation"] = False
    return conf


# ======================================================================================
# COMMAND LINE INTERFACE
# ======================================================================================
def trainer_cls() -> type:
    from raylab.agents.sac import SACTrainer
    from vmac.agents.mixins import CheckpointArtifactMixin

    class Trainable(CheckpointArtifactMixin, SACTrainer):
        pass

    return Trainable


@click.command()
@click.option(
    "--env",
    type=str,
    required=True,
)
@click.option(
    "--seed",
    type=int,
    default=0,
    help="Random number generator seed for random, numpy, and torch.",
)
@click.option(
    "--warmup",
    type=int,
    default=10000,
    show_default=True,
)
@click.option(
    "--evaluation-interval",
    type=int,
    default=10,
    show_default=True,
)
@tune_options
def main(tune_kwargs: dict, **options):
    import ray
    from ray import tune
    import raylab

    ray.init()
    raylab.register_all()

    name = "SAC"
    trainable_cls = trainer_cls()
    config = get_config(**options)

    tune_defaults(name, config, tune_kwargs)
    tune.run(trainable_cls, config=config, **tune_kwargs)


if __name__ == "__main__":
    main()
