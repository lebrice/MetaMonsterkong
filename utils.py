import numpy as np
import os, shutil

def extract_sub_demos(state_traj_lst, action_traj_lst, sub_length):
	"""Assumes sub_length < min length of all demos
	"""
	new_state_traj_lst = []
	new_action_traj_lst = []
	
	n_demos = len(state_traj_lst)
	avg_length = sum([len(traj) for traj in state_traj_lst])/n_demos
	n_sub_demos = int(n_demos*avg_length/sub_length)

	for _ in range(n_sub_demos):
		idx = np.random.randint(n_demos)
		len_demo = len(state_traj_lst[idx])
		start_idx = np.random.randint(len_demo - sub_length)
		new_state_traj_lst.append(state_traj_lst[idx][start_idx:start_idx+sub_length])
		new_action_traj_lst.append(action_traj_lst[idx][start_idx:start_idx+sub_length])
		
	return np.array(new_state_traj_lst), np.array(new_action_traj_lst)

def extract_sub_demos_sweep(state_traj_lst, action_traj_lst, sub_length):
	"""Assumes sub_length < min length of all demos
	"""
	new_state_traj_lst = []
	new_action_traj_lst = []
	
	n_demos = len(state_traj_lst)

	for idx in range(n_demos):
		len_demo = len(state_traj_lst[idx])
		for start_idx in range(len_demo - sub_length + 1):
			new_state_traj_lst.append(state_traj_lst[idx][start_idx:start_idx+sub_length])
			new_action_traj_lst.append(action_traj_lst[idx][start_idx:start_idx+sub_length])
		
	return np.array(new_state_traj_lst), np.array(new_action_traj_lst)

def clean_and_makedirs(dir_name, exp_name, seed, evaluate=False):
	main_dir = f'./{dir_name}/{exp_name}/seed_{seed}'
	save_path = os.path.join(main_dir, 'weights')
	eval_path = os.path.join(main_dir, 'eval')
	log_path = os.path.join(main_dir, 'log')
	
	if not evaluate: 
		if os.path.exists(main_dir):
			shutil.rmtree(main_dir)
		os.makedirs(save_path, exist_ok=True)
		os.makedirs(eval_path, exist_ok=True)
		os.makedirs(log_path, exist_ok=True)
	else:
		if os.path.exists(eval_path):
			shutil.rmtree(eval_path)
			os.makedirs(eval_path, exist_ok=True)

	return save_path, eval_path, log_path

def kld_gauss(mean_1, std_1, mean_2, std_2):
	kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
				(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
				std_2.pow(2) - 1)
	return	0.5 * torch.mean(torch.sum(kld_element, dim=1))