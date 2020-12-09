import os

import gym
from gym import Env
import numpy as np
import torch
from gym.spaces.box import Box

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.vec_normalize import \
	VecNormalize as VecNormalize_

from baselines.common.vec_env import (
    VecExtractDictObs,
    VecEnvObservationWrapper,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)

class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.
        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)

class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].
    Optionally normalize observations and scale reward.
    """

    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

class EpsilonGreedyWrapper(gym.Wrapper):
	"""
	Wrapper to perform a random action each step instead of the requested action, 
	with the provided probability.
	"""
	def __init__(self, env, prob=0.05):
		gym.Wrapper.__init__(self, env)
		self.prob = prob
		self.num_envs = env.num_envs

	def reset(self):
		return self.env.reset()

	def step(self, action):
		if np.random.uniform()<self.prob:
			action = np.random.randint(self.env.action_space.n, size=self.num_envs)
		
		return self.env.step(action)	

def make_vec_envs(envs,
				  gamma,                 
				  device,
				  num_frame_stack=4,
				  ob=False,
				  normalize=True,
				  options=None):

	if options == 'extract_rgb':
		envs = VecExtractDictObs(envs, "rgb")				

	envs = VecMonitor(venv=envs, filename=None, keep_buf=100)

	if normalize:
		if gamma is None:
			envs = VecNormalize(envs, ret=False, ob=ob)
		else:
			envs = VecNormalize(envs, gamma=gamma, ob=ob)

	envs = VecPyTorch(envs, device)

	if num_frame_stack is not None:
		envs = VecPyTorchFrameStack(envs, num_frame_stack, device)			

	return envs

class TransposeObs(gym.ObservationWrapper):
	def __init__(self, env=None):
		"""
		Transpose observation space (base class)
		"""
		super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
	def __init__(self, env=None, op=[2, 0, 1]):
		"""
		Transpose observation space for images
		"""
		super(TransposeImage, self).__init__(env)
		assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
		self.op = op
		obs_shape = self.observation_space.shape
		self.observation_space = Box(
			self.observation_space.low[0, 0, 0],
			self.observation_space.high[0, 0, 0], [
				obs_shape[self.op[0]], obs_shape[self.op[1]],
				obs_shape[self.op[2]]
			],
			dtype=self.observation_space.dtype)

	def observation(self, ob):
		return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
	def __init__(self, venv, device):
		"""Return only every `skip`-th frame"""
		super(VecPyTorch, self).__init__(venv)
		self.device = device
		self.spec = None
		# TODO: Fix data types

	def reset(self):
		obs = self.venv.reset()
		obs = torch.from_numpy(obs).float().to(self.device)
		return obs

	def step_async(self, actions):
		if isinstance(actions, torch.LongTensor) or isinstance(actions, torch.cuda.LongTensor):
			# Squeeze the dimension for discrete actions
			actions = actions.squeeze(1)
		actions = actions.cpu().numpy()
		self.venv.step_async(actions)

	def step_wait(self):
		obs, reward, done, info = self.venv.step_wait()
		obs = torch.from_numpy(obs).float().to(self.device)
		reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
		return obs, reward, done, info	


class VecNormalize(VecNormalize_):
	def __init__(self, *args, **kwargs):
		super(VecNormalize, self).__init__(*args, **kwargs)
		self.training = True

	def _obfilt(self, obs):
		if self.ob_rms:
			if self.training:
				self.ob_rms.update(obs)
			obs = np.clip((obs - self.ob_rms.mean) /
						  np.sqrt(self.ob_rms.var + self.epsilon),
						  -self.clipob, self.clipob)
			return obs
		else:
			return obs

	def train(self):
		self.training = True

	def eval(self):
		self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')

        self.device = device

        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape).to(self.device)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)