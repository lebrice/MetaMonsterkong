import gym
from gym.spaces.box import Box
from gym.wrappers.time_limit import TimeLimit
from gym_ple.ple_env import PLEEnv

from .make_env import make_env, make_vec_random_env

from gym.envs.registration import registry, register, make, spec
from gym_ple.ple_env import PLEEnv

# TODO: Change this if we add randomness to the envs.
nondeterministic = False
register(
    id="MetaMonsterKong-v0",
    entry_point="meta_monsterkong.make_env:MetaMonsterKongEnv",
    nondeterministic=nondeterministic,
)
register(
    id="MetaMonsterKong-v1",
    entry_point="meta_monsterkong.make_env:make_env",
    nondeterministic=nondeterministic,
)