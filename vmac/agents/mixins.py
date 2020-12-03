# pylint:disable=missing-docstring
import json
import os.path as osp

import numpy as np
import wandb
from ray.tune.logger import _SafeFallbackEncoder
from raylab.policy.off_policy import OffPolicyMixin


# ======================================================================================
# LOG TRAINER CHECKPOINT AND CONFIG AS ARTIFACT
# ======================================================================================
class CheckpointArtifactMixin:  # pylint:disable=too-few-public-methods
    def save(self, checkpoint_dir=None):
        checkpoint_path = super().save(checkpoint_dir)

        if self.wandb.enabled:
            artifact = wandb.Artifact(
                f"{self._name}_{self._env_id}_{self.iteration}its",
                type="dataset",
            )

            checkpoint_dir = osp.dirname(checkpoint_path)
            replay_path = self._save_replay(checkpoint_dir)
            if replay_path:
                artifact.add_file(replay_path)
            artifact.add_file(checkpoint_path, name="checkpoint")
            artifact.add_file(
                checkpoint_path + ".tune_metadata", name="checkpoint.tune_metadata"
            )
            artifact.add_file(self._save_config(checkpoint_dir))

            self.wandb.run.log_artifact(artifact)
        return checkpoint_path

    def _save_replay(self, checkpoint_dir: str) -> str:
        npz_path = osp.join(checkpoint_dir, "replay.npz")
        policy = self.get_policy()
        if isinstance(policy, OffPolicyMixin):
            replay = self.get_policy().replay
            # pylint:disable=protected-access
            storage = {k: v[: len(replay)] for k, v in replay._storage.items()}
            np.savez(npz_path, **storage)
            return npz_path

        return None

    def _save_config(self, checkpoint_dir: str) -> str:
        json_path = osp.join(checkpoint_dir, "config.json")
        with open(json_path, "w") as wfp:
            json.dump(
                self.config, wfp, indent=2, sort_keys=True, cls=_SafeFallbackEncoder
            )
        return json_path
