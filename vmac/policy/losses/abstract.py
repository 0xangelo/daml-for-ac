# pylint:disable=missing-module-docstring
from abc import ABCMeta
from typing import Tuple

from raylab.policy.losses import Loss as Base
from raylab.utils.dictionaries import get_keys
from raylab.utils.types import TensorDict
from torch import Tensor


class Loss(Base, metaclass=ABCMeta):  # pylint:disable=missing-class-docstring
    batch_keys: Tuple[str, ...]

    def unpack_batch(self, batch: TensorDict) -> Tuple[Tensor, ...]:
        """Returns the batch tensors corresponding to the batch keys.

        Tensors are returned in the same order `batch_keys` is defined.

        Args:
            batch: Dictionary of input tensors

        Returns:
            A tuple of tensors corresponding to each key in `batch_keys`
        """
        return tuple(get_keys(batch, *self.batch_keys))
