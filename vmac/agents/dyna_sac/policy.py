"""Policy for Dyna-SAC in PyTorch."""
from typing import List
from typing import Tuple

from ray.rllib.utils import PiecewiseSchedule
from raylab.agents.sac import SACTorchPolicy
from raylab.options import configure
from raylab.options import option
from raylab.policy.losses import DynaQLearning as NewActDynaQ
from raylab.policy.model_based import EnvFnMixin
from raylab.policy.model_based.lightning import LightningModelTrainer
from raylab.policy.model_based.lightning import TrainingSpec
from raylab.policy.model_based.policy import MBPolicyMixin
from raylab.policy.model_based.policy import model_based_options
from raylab.policy.modules.critic import SoftValue
from raylab.torch.optim.utils import build_optimizer
from raylab.utils.types import StatDict

from vmac.policy.losses import DynaQLearning as OldActDynaQ
from vmac.policy.losses import VAML


def default_model_training() -> dict:
    # pylint:disable=missing-function-docstring
    spec = TrainingSpec()
    spec.datamodule.holdout_ratio = 0.2
    spec.datamodule.max_holdout = None
    spec.datamodule.batch_size = 256
    spec.datamodule.shuffle = True
    spec.datamodule.num_workers = 0
    spec.training.max_epochs = None
    spec.training.max_steps = 120
    spec.training.patience = None
    spec.warmup = spec.training
    return spec.to_dict()


@configure
@model_based_options
@option("losses/", help="Options for model loss function")
@option("losses/grad_estimator", default="PD", help="One of {'PD', 'SF'}")
@option(
    "losses/model_samples", default=10, help="Positive number of next-state samples"
)
@option(
    "losses/nll_schedule",
    default=[(0, 0.001)],
    help="Negative log-likelihood interpolation factor in VAML",
)
@option(
    "losses/old_actions",
    default=False,
    help="""\
    Whether to use actions from the replay buffer or sample from the current policy.

    Refers to the actions used for the current state in Dyna Q-Learning.
    """,
)
@option("module/type", default="ModelBasedSAC", override=True)
@option("optimizer/models", default=dict(type="Adam", lr=1e-4))
@option("model_training", default=default_model_training())
@option("evaluation_config/explore", default=False, override=True)
class DynaSACTorchPolicy(MBPolicyMixin, EnvFnMixin, SACTorchPolicy):
    """Model-based policy por Dyna-SAC."""

    # pylint:disable=too-many-ancestors
    nll_schedule: PiecewiseSchedule

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_model_loss()
        self._set_nll_schedule()
        self.build_timers()

        self._grad_step = 0
        self.model_trainer = LightningModelTrainer(
            models=self.module.models,
            loss_fn=self.loss_model,
            optimizer=self.optimizers["models"],
            replay=self.replay,
            config=self.config,
        )

    def _setup_model_loss(self):
        options = self.config["losses"]
        self.loss_model = VAML(self.module.models, self._get_soft_value())
        self.loss_model.grad_estimator = options["grad_estimator"]
        self.loss_model.model_samples = options["model_samples"]

    def _setup_critic_loss(self):
        if self.config["losses"]["old_actions"]:
            self.loss_critic = OldActDynaQ(
                critics=self.module.critics,
                target_critic=self._get_soft_value(),
                models=self.module.models,
            )
        else:
            self.loss_critic = NewActDynaQ(
                self.module.critics,
                self.module.actor,
                self.module.models,
                self._get_soft_value(),
            )
        self.loss_critic.gamma = self.config["gamma"]
        self.loss_critic.model_samples = self.config["losses"]["model_samples"]
        self.loss_critic.seed(self.config["seed"])

    def _get_soft_value(self) -> SoftValue:
        return SoftValue(
            self.module.actor,
            self.module.target_critics,
            self.module.alpha,
            deterministic=False,
        )

    def _set_nll_schedule(self):
        schedule = self.config["losses"]["nll_schedule"]
        self.nll_schedule = PiecewiseSchedule(
            schedule,
            framework="torch",
            outside_value=schedule[-1][-1],
        )

    def compile(self):
        self.loss_model.compile()
        self.loss_actor.compile()
        self.loss_critic.compile()
        self.loss_alpha.compile()

    def _make_optimizers(self):
        optimizers = super()._make_optimizers()
        optimizers["models"] = build_optimizer(
            self.module.models, config=self.config["optimizer"]["models"]
        )
        return optimizers

    def train_dynamics_model(
        self, warmup: bool = False
    ) -> Tuple[List[float], StatDict]:
        self.loss_model.alpha = self.nll_schedule(self.global_timestep)
        return self.model_trainer.optimize(warmup=warmup)

    def _set_reward_hook(self):
        self.loss_critic.set_reward_fn(self.reward_fn)

    def _set_termination_hook(self):
        self.loss_critic.set_termination_fn(self.termination_fn)
