# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import random
import numpy as np
import collections
from gym.core import Env

Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])

def get_random_state_array(rand_state):
    """Creates a custom random state numpy array dtype for storing numpy random states.

    Args:
        rand_state: the random state tuple of a `RandomState` object

    Returns:
        The custom dtype object, its size, and an array of that dtype
    """
    dtype_list = []
    rand_state_list = []
    rand_state_array_size = 0
    for i in range(len(rand_state[1])):
        dtype_list.append(('unsigned integer key '+str(i), type(rand_state[1][0])))
        rand_state_list.append(rand_state[1][i])
    rand_state_array_size += len(np.asarray(rand_state[1][0]).tobytes())*len(rand_state[1])
    dtype_list.append(('pos', type(rand_state[2])))
    rand_state_list.append(rand_state[2])
    rand_state_array_size += len(np.asarray(rand_state[2]).tobytes())
    dtype_list.append(('has_gauss', type(rand_state[3])))
    rand_state_list.append(rand_state[3])
    rand_state_array_size += len(np.asarray(rand_state[3]).tobytes())
    dtype_list.append(('cached_gaussian', type(rand_state[4])))
    rand_state_list.append(rand_state[4])
    rand_state_array_size += len(np.asarray(rand_state[4]).tobytes())

    rand_state_array_type = np.dtype(dtype_list)

    rand_state_array = np.asarray([tuple(rand_state_list)], dtype=rand_state_array_type)

    return rand_state_array_type, rand_state_array_size, rand_state_array


def get_random_state_from_array(rand_state_array):
    """Obtains the random state tuple from a custom `rand_state_array_type`

    Args:
        rand_state_array: an object of numpy array dtype `rand_state_array_type`

    Returns:
        a numpy RandomState object with state set according to rand_state_array
    """
    rand_state_tuple = (
        'MT19937',
        np.array(list(rand_state_array[0])[:len(np.random.random.__self__.get_state()[1])]),
        rand_state_array[0]['pos'],
        rand_state_array[0]['has_gauss'],
        rand_state_array[0]['cached_gaussian']
    )

    return rand_state_tuple


def tf_set_seeds(seed):
    """Sets tensorflow seed.

    It is important to note that this seed affects the current default graph only.
    If you have another graph within the same session, you have to set the random seed
    within the scope of the "with graph.as_default()" block.

    Args:
        seed: int value for the seed
    """
    import tensorflow as tf
    tf.set_random_seed(seed)
    random.seed(seed)


class EnvSpec():
    def __init__(self, env_spec, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space
        self._unwrapped_spec = env_spec

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


class NormalizedEnv(Env):
    def __init__(
            self,
            env,
            scale_reward=1.,
            normalize_obs=False,
            normalize_reward=False,
            obs_alpha=0.001,
            reward_alpha=0.001,
    ):
        self._wrapped_env = env
        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(env.observation_space.shape)
        self._obs_var = np.ones(env.observation_space.shape)
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.
        self._spec = EnvSpec(env.spec, self.observation_space, self.action_space)

    def _update_obs_estimate(self, obs):
        flat_obs = self.wrapped_env.observation_space.flatten(obs)
        self._obs_mean = (1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(flat_obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        self._reward_mean = (1 - self._reward_alpha) * self._reward_mean + self._reward_alpha * reward
        self._reward_var = (1 - self._reward_alpha) * self._reward_var + \
                           self._reward_alpha * np.square(reward - self._reward_mean)

    def _apply_normalize_obs(self, obs):
        self._update_obs_estimate(obs)
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    def reset(self):
        ret = self._wrapped_env.reset()
        if self._normalize_obs:
            return self._apply_normalize_obs(ret)
        else:
            return ret

    def step(self, action):
        # rescale the action
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)
        return Step(next_obs, reward * self._scale_reward, done, info)
        # return next_obs, reward * self._scale_reward, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def start(self):
        return self._wrapped_env.start()

    def close(self):
        super(NormalizedEnv, self).close()
        return self._wrapped_env.close()

    @property
    def action_space(self):
        return self._wrapped_env.action_space

    @property
    def observation_space(self):
        return self._wrapped_env.observation_space

    def render(self, *args, **kwargs):
        try:
            return self._wrapped_env.render(*args, **kwargs)
        except TypeError:
            pass
            # return self._wrapped_env.render()

    def log_diagnostics(self, paths, *args, **kwargs):
        self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        self._wrapped_env.terminate()

    def get_param_values(self):
        return self._wrapped_env.get_param_values()

    def set_param_values(self, params):
        self._wrapped_env.set_param_values(params)

    def __getattr__(self, attr):
        orig_attr = self.wrapped_env.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.wrapped_env:
                    return self
                return result

            return hooked
        else:
            return orig_attr