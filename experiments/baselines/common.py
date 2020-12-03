# pylint:disable=missing-docstring
def wandb_config(file: str) -> dict:
    import os.path as osp

    config = {
        "file_paths": [osp.abspath(file)],
        "save_checkpoints": True,
        # wandb.init kwargs
        "project": "baselines",
        "entity": "angelovtt",
        "tags": ["baseline"],
    }
    return config


# ======================================================================================
# COMMON SETUPS
# ======================================================================================
def tune_defaults(trainable, config, tune_kwargs):
    if not tune_kwargs["checkpoint_freq"]:
        tune_kwargs["checkpoint_freq"] = config["evaluation_interval"]
    if not tune_kwargs["name"]:
        tune_kwargs["name"] = trainable
    if not tune_kwargs["stop"]:
        tune_kwargs["stop"] = {"timesteps_total": config["policy"]["buffer_size"]}
