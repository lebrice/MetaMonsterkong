""" WIP: Adding all things needed for us to be able to use meta-MonsterKong within
sequoia here.
"""
import os
from pathlib import Path
from typing import Optional

import gym
import numpy as np
from gym import Space
from gym_ple import PLEEnv
from meta_monsterkong.make_env import MetaMonsterKongEnv, MkConfig, make_env

source_dir = Path(os.path.dirname(os.path.abspath(__file__)))


def make_monsterkong_task_env(task_id: int, mode: str = "train") -> MetaMonsterKongEnv:
    """Create a 'monsterkong' environment associated with the given 'task', for use within
    [Sequoia](https://www.github.com/lebrice/Sequoia).

    TODO: Need to decide how the 'tasks' and 'mode' changes the envs:
    - map the task ids to the map files?
    - change textures for train / valid / test?
    - Different set of levels for train / valid / test?

    Parameters
    ----------
    task_id : int
        The index of a 'task'. The environment should be consistent when the same task
        is given twice.

    mode : str
        Which mode to use, one of 'train', 'valid' or 'test'. Defaults to 'train'.

    Returns
    -------
    PLEEnv
        A PyGame learning environment for MonsterKong, configured for the given task.
    """
    raise NotImplementedError("TODO")

    if mode not in {"train", "valid", "test"}:
        raise RuntimeError(f"mode should be one of 'train', 'valid', or 'test'.")
    # TODO: Change this mk_config depending on train / valid / test
    mk_config = MkConfig(
        MapHeightInTiles=20,
        MapWidthInTiles=20,
        IsRender=True,
        SingleID=None,
        DenseRewardCoeff=0.0,
        RewardsWin=50.0,
        StartLevel=task_id,
        NumLevels=1,
        TextureFixed=True,  # False,
        Mode="Test",
    )
    return make_env(mk_config=mk_config)


class RandomAgent:
    """Just for illustration purposes, tryign to get to something close to Sequoia's
    Method class.
    """

    def get_actions(self, observation: np.ndarray, action_space: Space):
        return action_space.sample()

    def fit(self, train_env: PLEEnv, valid_env: PLEEnv):
        """Fake training method

        (just here for illustration purposes)

        Parameters
        ----------
        train_env : [type]
            Environment used for training.
        valid_env : [type]
            Environment used for validation.
        """

    def on_task_switch(self, task_id: Optional[int]):
        """Called when switching between tasks.

        (just here for illustration purposes)

        """


class ExampleMonsterKongSetting:
    """Example of how we could create a Setting for MonsterKong.
    (Not really used in Sequoia, just here for illustration purposes)

    In reality though, we'll just add meta-monsterkong as one potential environment to
    be used by the RL settings already implemented in Sequoia.
    """

    def __init__(
        self,
        nb_tasks: int = 5,
        n_episodes: int = 3,
        batch_size: Optional[int] = None,
        render: bool = False,
    ):
        self.nb_tasks = nb_tasks
        self.n_episodes = n_episodes
        self.render = render
        self.batch_size: Optional[int] = batch_size 
        # Get the observation / action space (doesn't change across tasks)
        with MetaMonsterKongEnv() as temp_env:
            self.observation_space = temp_env.observation_space
            self.action_space = temp_env.action_space

    def apply(self, method):
        """Fake 'apply' method, as an example to illustrate how Sequoia would do it."""
        transfer_matrix = np.zeros([self.nb_tasks, self.nb_tasks], dtype=float)
        train_env = MetaMonsterKongEnv()
        train_env.seed(123)
        valid_env = MetaMonsterKongEnv()
        train_env.seed(456)
        test_env = MetaMonsterKongEnv()
        test_env.seed(999)
        
        for task_id in range(self.nb_tasks):
            print(f"Starting Training on task {task_id}")
            train_env.set_task(level=task_id,color="RG")
            valid_env.set_task(level=task_id,color="RB")
            
            method.on_task_switch(task_id)
            method.fit(train_env, valid_env)

            for test_task_id in range(self.nb_tasks):
                print(f"Testing on task {test_task_id}")
                test_env.set_task(level=test_task_id,color="GB")

                assert test_env.get_level() == test_task_id

                method.on_task_switch(test_task_id)
                task_performance = self.test(method, test_env)

                test_env.close()

                transfer_matrix[task_id][test_task_id] = task_performance

        print("Results: ")
        # Normally you'd create some Results object and return that:
        # objective = some_function(transfer_matrix)
        # results = Results(transfer_matrix=transfer_matrix)
        print(transfer_matrix)
        return transfer_matrix

    def test(self, agent: RandomAgent, test_env) -> float:
        """Evaluate the performance of an agent on the given env.

        (just here for illustration / experimentation purposes)

        Parameters
        ----------
        agent : RandomAgent
            [description]
        env : PLEEnv
            [description]

        Returns
        -------
        float
            [description]
        """
        reward = 0
        done = False
        rewards = []
        for i in range(self.n_episodes):
            print(f"Episode {i} / {self.n_episodes}")
            episode_rewards = []
            ob = test_env.reset()
            while True:
                if self.render:
                    test_env.render(mode="human")
                action = agent.get_actions(ob, self.action_space)
                ob, reward, done, _ = test_env.step(action)
                episode_rewards.append(reward)
                if done:
                    break
            total_reward = sum(episode_rewards)
            rewards.append(total_reward)
        mean_reward = np.array(rewards).mean()
        print(f"average={mean_reward}")
        return mean_reward


if __name__ == "__main__":
    method = RandomAgent()
    setting = ExampleMonsterKongSetting(nb_tasks=5, render=True)
    results = setting.apply(method)
    print(f"Results: {results}")
