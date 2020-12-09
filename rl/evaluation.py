import numpy as np
import torch

from . import utils


def evaluate(actor_critic, ob_rms, eval_envs, device, writer, epoch, config, render=False):
    
    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        1, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(1, 1, device=device)

    print("Evaluating Envs")
    while len(eval_episode_rewards) < config['num_eval']:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Observe reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        if render:
            eval_envs.render()

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])              

    print(" Evaluation using {} episodes: mean reward {:.5f} \n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
            
    writer.add_scalar("mean reward", np.mean(eval_episode_rewards), epoch)
    writer.add_scalar("median reward", np.median(eval_episode_rewards), epoch)
    writer.add_scalar("max reward", np.max(eval_episode_rewards), epoch)
    writer.add_scalar("min reward", np.min(eval_episode_rewards), epoch)