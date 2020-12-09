import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.feature_extractors import ImpalaCNN, NatureDQNCNN
from models.policies import Policy, NNBase
from models.distributions import Categorical
from envs.meta_monsterkong import monsterkong_randomensemble

from rl import algo, utils
from utils import clean_and_makedirs
from rl.wrappers import make_vec_envs
from gym import wrappers as gym_wrappers
from rl.storage import RolloutStorage
from rl.evaluation import evaluate
import json
from torch.utils.tensorboard import SummaryWriter
import argparse

FEAT_DICT = {
    'impala': ImpalaCNN,
    'nature_dqn': NatureDQNCNN,
}


def main(config):
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.set_num_threads(1)
    np.random.seed(config['seed'])

    is_cuda = config['cuda'] and torch.cuda.is_available()

    if is_cuda and config['cuda_deterministic']:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    dir_name = 'rl_experiments/monsterkong'    
    save_dir, eval_dir, log_dir = clean_and_makedirs(dir_name=dir_name, exp_name=config['exp_name'], seed=config['seed'])
    summary_writer = SummaryWriter(log_dir=log_dir)
    eval_writer = SummaryWriter(log_dir=eval_dir)

    device_str = "cuda" if is_cuda else "cpu"
    device = torch.device(device_str)

    mk_config_train = {
        'MapsDir':'./envs/meta_monsterkong/monsterkong_randomensemble/maps20x20',
        'MapHeightInTiles': 20,
        'MapWidthInTiles': 20,
        'IsRender':True,
        'RewardsWin':50.0,
        'StartLevel':config['start_lev'],
        'NumLevels':config['n_lev'],
        'TextureFixed':config['texture_fixed'],
        'Mode': 'Train',    
    }

    mk_config_test = {
        'MapsDir':'./envs/meta_monsterkong/monsterkong_randomensemble/maps20x20_test',
        'MapHeightInTiles': 20,
        'MapWidthInTiles': 20,
        'IsRender':True,
        'RewardsWin':50.0,
        'StartLevel':0,
        'NumLevels':1000,
        'TextureFixed':config['texture_fixed'],
        'Mode': 'Test',    
    }    

    envs = monsterkong_randomensemble.make_vec_random_env(num_envs=config['num_processes'], mk_config=mk_config_train)    
    envs = make_vec_envs(envs=envs, gamma=config['gamma'], device=device, **config['vec_env_kwargs'])
    
    eval_envs = monsterkong_randomensemble.make_vec_random_env(num_envs=1, mk_config=mk_config_test)
    eval_envs = make_vec_envs(envs=eval_envs, gamma=config['gamma'], device=device, **config['vec_env_kwargs']) 

    feat_model = FEAT_DICT[config['feat_choice']]
    feat_model = feat_model(**config['feat_args'])
    feat_model.to(device)

    act_dim = envs.action_space.n

    actor_critic = Policy(
        base = NNBase(
            feat_model = feat_model, 
            act_dist_cls = Categorical, 
            act_dim = act_dim,
            is_recurrent = False,
        ),  
    )
    action_space = envs.action_space  


    actor_critic.to(device)
    actor_critic.eval()

    if config['load_path']:
        load_file = torch.load(config['load_path'], map_location=torch.device(device))
        actor_critic.load_state_dict(load_file['actor_critic'])

        ob_rms = load_file['ob_rms']
        vec_norm = utils.get_vec_normalize(envs)
        
        if vec_norm is not None:
            vec_norm.ob_rms = ob_rms        

    agent = algo.PPO(
        actor_critic,
        config['clip_param'],
        config['ppo_epoch'],
        config['num_mini_batch'],
        config['value_loss_coef'],
        config['entropy_coef'],
        lr=config['lr'],
        eps=config['eps'],
        max_grad_norm=config['max_grad_norm']
    )

    rollouts = RolloutStorage(config['num_steps'], config['num_processes'],
                              envs.observation_space.shape, action_space,
                              actor_critic.recurrent_hidden_state_size)

    vec_norm = utils.get_vec_normalize(envs)
    if vec_norm is not None:
        vec_norm.train()
    
    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()

    obs = envs.reset()

    if config['blind']:
        obs = 0*obs

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=config["epinfo_buf_len"])

    start = time.time()
    num_updates = int(
        config['num_env_steps']) // config['num_steps'] // config['num_processes']

    for j in range(num_updates):

        if config['use_linear_lr_decay']:
            # decrease learning rate linearly
            utils.update_linear_lr_schedule(agent.optimizer, j, num_updates, config['lr'])

        if config['use_linear_clip_decay']:
            # decrease learning rate linearly
            utils.update_linear_clip_schedule(agent, j, num_updates, config['clip_param'])

        for step in range(config['num_steps']):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Observe reward and next obs           
            obs, reward, done, infos = envs.step(action)  

            if config['blind']:
                obs = 0*obs

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, config['use_gae'], config['gamma'],
                                 config['gae_lambda'], config['time_limits'])

        actor_critic.train()

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        actor_critic.eval()

        # save for every interval-th episode or for the last epoch
        if (j % config['save_interval'] == 0) or (j == num_updates - 1):
            save_path = os.path.join(save_dir, config['algo'])
            os.makedirs(save_path, exist_ok=True)
            save_dict = {'actor_critic': actor_critic.state_dict(), 
                         'ob_rms': getattr(utils.get_vec_normalize(envs), 'ob_rms', None)}

            torch.save(save_dict, os.path.join(save_path, f"{config['exp_name']}.pt"))

        if j % config['log_interval'] == 0 and (len(episode_rewards) == config["epinfo_buf_len"]):
            total_num_steps = (j + 1) * config['num_processes'] * config['num_steps']
            
            end = time.time()

            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.5f}/{:.5f}, min/max reward {:.5f}/{:.5f} \n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))   

            summary_writer.add_scalar("mean reward", np.mean(episode_rewards), total_num_steps)
            summary_writer.add_scalar("median reward", np.median(episode_rewards), total_num_steps)
            summary_writer.add_scalar("max reward", np.max(episode_rewards), total_num_steps)
            summary_writer.add_scalar("min reward", np.min(episode_rewards), total_num_steps)
            summary_writer.add_scalar("distribution entropy", dist_entropy, total_num_steps)
            summary_writer.add_scalar("value loss", value_loss, total_num_steps)
            summary_writer.add_scalar("action loss", action_loss, total_num_steps)           

        if (config['eval_interval'] > 0) and (j % config['eval_interval'] == 0) and (len(episode_rewards) == config["epinfo_buf_len"]):
            ob_rms = getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            evaluate(actor_critic, ob_rms, eval_envs, device_str, eval_writer, j, config)
    
    envs.close()
    eval_envs.close()


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Config')  
    parser.add_argument('--feat_choice', type=str, default='impala', choices=FEAT_DICT.keys(), help='feature choice')
    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--start_lev', type=int, default=0, help='start level')
    parser.add_argument('--n_lev', type=int, default=200, help='num levels')    
    parser.add_argument('--blind', action='store_true', help='no image available')
    parser.add_argument('--texture_fixed', action='store_true', help='no image available')    
    parser.add_argument('--suffix', type=str, default=None, help='suffix of the name')
    parser.add_argument('--load_path', type=str, default=None, help='eval path')

    args = parser.parse_args()

    config_file = f'./rl_config/monsterkong/monsterkong_{args.feat_choice}.json'
    
    with open(config_file) as f:
        config = json.load(f)

    if args.seed is not None:
        config['seed'] = args.seed
    
    config['start_lev'] = args.start_lev
    config['n_lev'] = args.n_lev

    config['load_path'] = args.load_path

    if args.n_lev > 0:
        config['exp_name'] += f'_nlev_{args.n_lev}'

    if args.start_lev > 0:
        config['exp_name'] += f'_slev_{args.start_lev}'        

    config['blind'] = args.blind
    config['texture_fixed'] = args.texture_fixed

    if args.blind:
        config['exp_name'] += '_blind'

    if args.texture_fixed:
        config['exp_name'] += '_texture_fixed'        

    if args.suffix:
        config['exp_name'] += f'_{args.suffix}'        

    config['exp_name'] += '_rl'
    
    print(config)
    main(config)
