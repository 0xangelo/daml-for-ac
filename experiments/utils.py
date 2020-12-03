# pylint:disable=missing-docstring
import itertools
import json
import logging
import os.path as osp
import typing as ta

import numpy as np
import wandb  # pylint:disable=wrong-import-order
from ray.rllib import SampleBatch
from raylab.agents import Trainer
from raylab.agents.registry import get_agent_cls
from raylab.agents.sac import SACTorchPolicy
from raylab.agents.sop import SOPTorchPolicy
from raylab.policy import TorchPolicy
from raylab.policy.model_based.lightning import DataModule
from raylab.policy.model_based.lightning import DatamoduleSpec
from raylab.policy.modules.actor import DeterministicPolicy
from raylab.policy.modules.actor import StochasticPolicy
from raylab.policy.modules.critic import HardValue
from raylab.policy.modules.critic import SoftValue
from raylab.policy.modules.critic import VValue
from raylab.utils.replay_buffer import NumpyReplayBuffer
from raylab.utils.replay_buffer import ReplayField
from wandb.wandb_run import Run


logger = logging.getLogger(__name__)


# ======================================================================================
# WANDB
# ======================================================================================


def wandb_run_initializer(*args, **kwargs) -> ta.Callable[[], Run]:
    def init_wandb(**options):
        run = wandb.init(*args, **kwargs)
        run.config.update(options)
        return run

    return init_wandb


# ======================================================================================
# ARTIFACT HANDLING
# ======================================================================================


def get_config_from_artifact(artifact: wandb.Artifact) -> dict:
    base_dir = artifact.download()
    with open(osp.join(base_dir, "config.json"), "r") as rfp:
        config = json.load(rfp)

    config.pop("callbacks", None)
    config.pop("wandb", None)
    config.pop("eager", None)
    config.pop("sample_batch_size", None)
    config.pop("use_pytorch", None)
    config.pop("model", None)
    return config


def get_agent_from_artifact(
    artifact: wandb.Artifact, agent_name: str = "SoftAC"
) -> Trainer:
    config = get_config_from_artifact(artifact)
    cls = custom_agent_cls(agent_name)
    agent = cls(config=config)

    agent.restore(osp.join(artifact.download(), "checkpoint"))
    return agent


def custom_agent_cls(agent_name: str) -> type:
    base = get_agent_cls(agent_name)

    # class Derived(base):
    #     def __setstate__(self, state):
    #         # Fix old checkpoints not having metrics
    #         state.setdefault("metrics", [0, 0])
    #         # Avoid restoring inexisting optimizer
    #         state.pop("optimizer", None)
    #         super().__setstate__(state)

    # return Derived
    return base


def get_policy_from_artifact(
    artifact: wandb.Artifact, agent_name: str = "SoftAC"
) -> TorchPolicy:
    agent = get_agent_from_artifact(artifact, agent_name=agent_name)
    policy = agent.get_policy()
    return policy


def get_value_from_policy(
    policy: ta.Union[SACTorchPolicy, SOPTorchPolicy], deterministic: bool = False
) -> VValue:
    model_free = policy.module
    if isinstance(model_free.actor, StochasticPolicy):
        return SoftValue(
            model_free.actor,
            model_free.target_critics,
            model_free.alpha,
            deterministic=deterministic,
        )
    if isinstance(model_free, DeterministicPolicy):
        return HardValue(model_free.target_actor, model_free.target_critics)

    raise ValueError(
        f"{type(policy).__name__} does not contain a valid actor cls:"
        f" {type(model_free.actor).__name__}"
    )


def single_precision_replay(replay: NumpyReplayBuffer) -> NumpyReplayBuffer:
    obs_field = act_field = None
    for field in replay.fields:
        if field.name in {SampleBatch.CUR_OBS, SampleBatch.NEXT_OBS}:
            obs_field = ReplayField(
                name=field.name, shape=field.shape, dtype=np.float32
            )
        elif field.name == SampleBatch.ACTIONS:
            act_field = field

    new = NumpyReplayBuffer(
        obs_field, act_field, replay._maxsize  # pylint:disable=protected-access
    )
    new.add_fields(
        *(f for f in replay.fields if f.name not in {n.name for n in new.fields})
    )
    return new


def restore_replay_from_artifact(artifact: wandb.Artifact, replay: NumpyReplayBuffer):
    with np.load(osp.join(artifact.download(), "replay.npz")) as chkpt:
        assert all([f.name in chkpt for f in replay.fields])
        replay.add(
            SampleBatch({f.name: chkpt[f.name].astype(f.dtype) for f in replay.fields})
        )


# ======================================================================================
# DATA MODULE
# ======================================================================================


def make_datamodule(replay: NumpyReplayBuffer, **kwargs) -> DataModule:
    spec = DatamoduleSpec(**kwargs)
    return DataModule(replay, spec)


# ======================================================================================
# GRID SEARCH
# ======================================================================================


def grid_kwargs(**options) -> ta.List[dict]:
    grid_keys = [k for k, v in options.items() if isinstance(v, tuple)]
    product = itertools.product(*[options[k] for k in grid_keys])
    grid_kwargs = [{k: v} for values in product for k, v in zip(grid_keys, values)]
    all_kwargs = [{**options, **kwargs} for kwargs in grid_kwargs]
    return all_kwargs
