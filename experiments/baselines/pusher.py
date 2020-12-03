# pylint:disable=missing-docstring
import click
from common import tune_defaults
from common import wandb_config
from raylab.cli.utils import tune_options
from sac import env_config
from sac import policy_config
from sac import trainer_cls
from sac import trainer_config


def get_config(seed: int, warmup: int, evaluation_interval: int) -> dict:
    config = {}
    config.update(pusher_config())
    config.update(trainer_config(seed, warmup, evaluation_interval))
    config.update(policy=custom_policy(warmup))
    config.update(wandb=wandb_config(__file__))
    return config


def pusher_config() -> dict:
    import gym

    env_id = "Pusher-v2"
    spec = gym.spec(env_id)
    cnf = env_config(env_id)
    cnf["env_config"]["time_aware"] = True
    cnf["env_config"]["max_episode_steps"] = spec.max_episode_steps
    return cnf


def custom_policy(warmup: int) -> int:
    from ray import tune

    cnf = policy_config(warmup)
    cnf["buffer_size"] = int(1e5)
    cnf["target_entropy"] = tune.grid_search(["auto", "tf-agents"])
    return cnf


@click.command()
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
