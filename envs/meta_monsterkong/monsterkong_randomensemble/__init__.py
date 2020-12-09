from gym_ple.ple_env import PLEEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from gym.wrappers.time_limit import TimeLimit
import gym
from gym.spaces.box import Box

class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

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

def make_vec_random_env(num_envs, mk_config):
	env_func = lambda: TransposeImage(TimeLimitMask(TimeLimit(PLEEnv(root_game_name="monsterkong_randomensemble", 
		game_name="MonsterKong_RandomEnsemble", display_screen=False, ple_game=False, reward_type='sparse', 
		obs_type='image', **{'mk_config': mk_config}), max_episode_steps=500)))
	
	if num_envs == 1:
		return DummyVecEnv([env_func for _ in range(num_envs)])

	return ShmemVecEnv([env_func for _ in range(num_envs)])         