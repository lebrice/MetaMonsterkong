import os
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Union

import gym
import numpy as np
from gym import spaces
from gym_ple import PLEEnv
from gym.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv
from ple import PLE

from dataclasses import dataclass, asdict, replace
from .wrappers import TimeLimit, TimeLimitMask, TransposeImage
from .envs import meta_monsterkong, MkConfig

source_dir = Path(os.path.dirname(os.path.abspath(__file__)))


default_config = MkConfig()

class MetaMonsterKongEnv(PLEEnv):
    def __init__(self,
                 mk_config: Union[MkConfig, Dict] = None,
                 observe_state: bool = False):
        if not mk_config:
            mk_config = default_config
        else:
            # Overwrite the defaults with the custom config, if passed.
            if isinstance(mk_config, MkConfig):
                mk_config = asdict(mk_config)
            mk_config = replace(default_config, **mk_config)
        self.mk_config: MkConfig = mk_config
        super().__init__(
            game_name="meta_monsterkong",
            display_screen=False,
            ple_game=False,
            root_game_name="meta_monsterkong",
            reward_type="sparse",
            obs_type="state" if observe_state else "image",
            mk_config=asdict(mk_config),
        )
        if isinstance(self.reward_range, (int, float)):
            # BUG: Reward range should be a tuple, not an int/float.
            assert self.reward_range > 0
            self.reward_range = (0, self.reward_range)

        self.observe_state = observe_state
        if self.observe_state:
            # TODO: Not sure exactly what the upper bound on the state space should be.
            self.observation_space = spaces.Box(0, 292, [402,], np.int16)
        else:
            self.observation_space = spaces.Box(0, 255, (64, 64, 3), np.uint8)
        
        assert self.reset() in self.observation_space, (self.reset(), self.observation_space)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    @property
    def game(self) -> meta_monsterkong:
        return self.ple_wrapper.game
    
    @property
    def level(self) -> Optional[int]:
        return self.game._level
    
    @level.setter
    def level(self, value: Optional[int]) -> None:
        self.game._level = int(value) if value is not None else value
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        # Small bug: done is an int rather than a boolean.
        if self.observe_state:
            # Cast to int16 to save a tiny bit of memory and match the above obs space.
            obs = obs.astype(np.int16)   
        return obs, reward, bool(done), info

    # NOTE: We also include these methods, so that we don't set the 'level' attribute on
    # a Wrapper by accident.

    def get_level(self) -> int:
        return self.game._level
    
    def set_level(self, value: int):
        self.game._level = value


def make_env(mk_config: Union[MkConfig, Dict] = None, observe_state: bool = False) -> PLEEnv:
    env = MetaMonsterKongEnv(mk_config=mk_config, observe_state=observe_state)
    # TODO: Do we want to always include these wrappers?
    env = TimeLimit(env, max_episode_steps=500)
    env = TimeLimitMask(env)
    env = TransposeImage(env)
    return env


def make_vec_random_env(num_envs: int, mk_config: Union[MkConfig, Dict]) -> VectorEnv:
    # Move import here in case we don't have `baselines` installed:
    # TODO: Use the "native" vectorized envs from gym rather than those from baselines.
    # The only thing we'd lose is the ability to render the envs, which isn't part of
    # gym at the time of writing. One potential solution would be to use a fork of gym
    # which adds this support for rendering the envs.
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv

    env_func = partial(make_env, mk_config=mk_config)

    if num_envs == 1:
        return DummyVecEnv([env_func for _ in range(num_envs)])
    return ShmemVecEnv([env_func for _ in range(num_envs)])
