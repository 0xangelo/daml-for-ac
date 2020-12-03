# pylint:disable=missing-docstring
import click
from raylab.cli.utils import tune_options


def get_config(env: str, seed: int, warmup: int, evaluation_interval: int) -> dict:
    config = {}
    config.update(env_config(env))
    config.update(trainer_config(warmup, evaluation_interval))
    config.update(policy_config(warmup))
    return config


def policy_config(warmup: int = 10000) -> dict:
    from ray import tune

    cnf = {
        "compile": True,
        "losses": {
            "grad_estimator": "PD",
            "model_samples": 10,
            "nll_schedule": [(0, 1.0)],
            "old_actions": tune.grid_search([True, False]),
            # "nll_schedule": tune.grid_search(
            #     [[(0, 1.0)], [(0, 1.0), (2001, 1.0), (2001, 0.001), (50000, 0.0)]]
            # ),
        },
        "module": module_config(),
        "optimizer": {
            "models": {"type": "Adam", "lr": 1e-4, "weight_decay": 1e-5},
            # Taken from pfrl
            "actor": {"type": "Adam", "lr": 3e-4},
            "critics": {"type": "Adam", "lr": 3e-4},
            "alpha": {"type": "Adam", "lr": 3e-4},
        },
        "polyak": 0.995,
        "target_entropy": "auto",
        "model_update_interval": 25,
        "buffer_size": int(1e5),
        "batch_size": 256,
        "improvement_steps": 1,
        "exploration_config": {"pure_exploration_steps": warmup},
    }
    return {"policy": cnf}


def module_config() -> dict:
    from raylab.policy.modules.mb_sac import ModelBasedSACSpec

    spec = ModelBasedSACSpec()
    spec.model.network.units = (128, 128)
    spec.model.network.activation = "Swish"
    spec.model.network.delay_action = False
    spec.model.network.standard_scaler = False
    spec.model.network.fix_logvar_bounds = True
    spec.model.network.input_dependent_scale = True
    spec.model.residual = True
    spec.model.initializer = {"name": "xavier_uniform"}
    spec.model.ensemble_size = 1

    # Same as model-free, taken from pfrl
    spec.actor.encoder.units = (256, 256)
    spec.actor.encoder.activation = "ReLU"
    spec.actor.encoder.layer_norm = False
    spec.actor.input_dependent_scale = True
    spec.actor.initial_entropy_coeff = 1.0
    spec.actor.initializer = {"name": "xavier_uniform"}

    spec.critic.encoder.units = (256, 256)
    spec.critic.encoder.activation = "ReLU"
    spec.critic.encoder.layer_norm = False
    spec.critic.double_q = True
    spec.critic.parallelize = True
    spec.critic.initializer = {"name": "xavier_uniform"}

    return {"type": "ModelBasedSAC", **spec.to_dict()}


def model_training_config() -> dict:
    from raylab.policy.model_based.lightning import TrainingSpec

    spec = TrainingSpec()
    spec.datamodule.holdout_ratio = 0.2
    spec.datamodule.max_holdout = 5000
    spec.datamodule.batch_size = 256
    spec.datamodule.shuffle = True
    spec.training.max_epochs = 20
    spec.training.max_steps = 200
    spec.training.patience = 5
    spec.training.improvement_delta = 0.0
    spec.warmup = spec.training.from_dict(spec.training.to_dict())
    spec.warmup.max_epochs = 1000
    spec.warmup.max_steps = None

    return {"model_training": spec.to_dict()}


def trainer_config(warmup: int = 10000, evaluation_interval: int = 10) -> dict:
    return {
        "gamma": 0.99,
        "timesteps_per_iteration": 200,
        "rollout_fragment_length": 1,
        "evaluation_interval": evaluation_interval,
        "evaluation_num_episodes": 10,
        "learning_starts": warmup,
        "evaluation_config": {"explore": False},
    }


def env_config(env: str) -> dict:
    conf = {
        "env": env,
        "env_config": {
            "time_aware": False,
            "max_episode_steps": 1000,
            "single_precision": True,
        },
    }
    if env.endswith("-v3"):
        kwargs = dict(exclude_current_positions_from_observation=False)
        conf["env_config"]["kwargs"] = kwargs
    return conf


# ======================================================================================
# COMMON SETUPS
# ======================================================================================
def wandb_config() -> dict:
    import os.path as osp

    config = {
        "file_paths": [osp.abspath(__file__)],
        "save_checkpoints": True,
        # wandb.init kwargs
        "project": "vaml",
        "entity": "angelovtt",
        "tags": ["NeurIPS"],
    }
    return config


def set_defaults(tune_kwargs: dict, trainable_name: str, config: dict):
    if not tune_kwargs["checkpoint_freq"]:
        tune_kwargs["checkpoint_freq"] = config["evaluation_interval"]
    if not tune_kwargs["name"]:
        tune_kwargs["name"] = trainable_name
    if not tune_kwargs["stop"]:
        tune_kwargs["stop"] = {"timesteps_total": config["buffer_size"]}


# ======================================================================================
# COMMAND LINE INTERFACE
# ======================================================================================
@click.command()
@click.option(
    "--env",
    type=str,
    default=None,
    show_default=True,
    help="Gym environment id (should be Mujoco)",
)
@click.option(
    "--seed",
    type=int,
    default=0,
    show_default=True,
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
def main(env: str, tune_kwargs: dict, **options):
    import ray
    from ray import tune
    import raylab
    from vmac.agents.dyna_sac import DynaSACTrainer
    from vmac.agents.mixins import CheckpointArtifactMixin

    ray.init()
    raylab.register_all()

    name = "Dyna-SAC"

    class Trainable(CheckpointArtifactMixin, DynaSACTrainer):
        # pylint:disable=too-many-ancestors
        _name = name

    env = env or tune.grid_search("InvertedPendulum-v2 Pusher-v2 Hopper-v3".split())
    config = get_config(env=env, **options)
    config["wandb"] = wandb_config()

    set_defaults(tune_kwargs, name, config)
    tune.run(Trainable, config=config, **tune_kwargs)


if __name__ == "__main__":
    main()  # pylint:disable=no-value-for-parameter
