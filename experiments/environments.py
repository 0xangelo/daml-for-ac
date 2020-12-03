# pylint:disable=missing-docstring
import gym
import torch

# ======================================================================================
# TimeAwareEnv
# ======================================================================================


class TimeAwareTransitionFn(gym.Wrapper):
    def __init__(self, env=None):
        super().__init__(env)

        _env = env
        while hasattr(_env, "env"):
            if isinstance(_env, gym.wrappers.TimeLimit):
                break
            _env = _env.env

        if hasattr(self.env, "transition_fn") and isinstance(
            _env, gym.wrappers.TimeLimit
        ):
            horizon = _env._max_episode_steps

            def transition_fn(state, action):
                state, rel_time = state[..., :-1], state[..., -1:]
                new_state, logp = self.env.transition_fn(state, action)

                time = torch.round(rel_time * horizon)
                new_rel_time = torch.clamp((time + 1) / horizon, 0, 1)
                new_state = torch.cat([new_state, new_rel_time], dim=-1)

                return new_state, logp

            self.transition_fn = transition_fn
