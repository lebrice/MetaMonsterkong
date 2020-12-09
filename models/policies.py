import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    """
    base: Policy Base. Outputs parameters of action distribution.
    Specfically, the inferface for base should be
        value, dist, rnn_hxs = self.base(obs, rnn_hxs, masks)
    action_dist: Samples actions based on the parameters.
    """
    def __init__(self, base):
        super(Policy, self).__init__()
        self.base = base

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.hidden_dim

    def forward(self, obs, rnn_hxs=None, masks=None, deterministic=False):
        # rnn_hxs and masks are only required for RNN policy
        return self.act(obs, rnn_hxs, masks, deterministic)

    def act(self, obs, rnn_hxs=None, masks=None, deterministic=False, act_dist=False):
        # rnn_hxs and masks are only required for RNN policy
        value, dist, rnn_hxs= self.base(obs, rnn_hxs, masks)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        if act_dist:
            return value, dist, action_log_probs, rnn_hxs

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, obs, rnn_hxs, masks):
        value, _, _ = self.base(obs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, obs, action, rnn_hxs=None, masks=None):
        value, dist, rnn_hxs = self.base(obs, rnn_hxs, masks)
        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def calc_nll_loss(self, obs, action, rnn_hxs=None, masks=None, logits=False):
        if logits:
            raise NotImplementedError
        else:
            _, action_log_probs, _, _ = self.evaluate_actions(obs, action, rnn_hxs, masks)
            return -action_log_probs.mean()

    def calc_mt_nll_loss(self, obs, action, rnn_hxs=None, masks=None, logits=False):
        if logits:
            raise NotImplementedError
        else:
            task_losses = []            
            obs_dim = obs.shape[-1]
            num_tasks = obs_dim - self.base.feat_model.state_dim
            _, tasks = obs[:, -num_tasks:].max(dim=1)
            for task_idx in range(num_tasks):
                idxs = torch.where(tasks == task_idx)[0]
                if len(idxs) == 0:
                    task_losses.append(None)
                else:
                    _, action_log_probs, _, _ = self.evaluate_actions(obs[idxs], action[idxs], rnn_hxs, masks)
                    task_losses.append(-action_log_probs.mean())
            return task_losses            

class NNBase(nn.Module):
    def __init__(self, feat_model, act_dist_cls, act_dim, is_recurrent=False, recurrent_hidden_dim=None):
        super(NNBase, self).__init__()
        self.feat_model = feat_model
        self.is_recurrent = is_recurrent
        self.action_dim = act_dim        
        self.hidden_dim = feat_model.feat_dim

        if is_recurrent:
            self.hidden_dim = recurrent_hidden_dim
            self.gru = nn.GRU(input_size=feat_model.feat_dim, hidden_size=recurrent_hidden_dim)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

        self.critic_linear = nn.Linear(self.hidden_dim, 1)

        self.action_dist = act_dist_cls(self.hidden_dim, act_dim)

    def forward(self, obs, rnn_hxs=None, masks=None):
        # rnn_hxs and masks are only required for RNN base
        x = self.feat_model(obs)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        value = self.critic_linear(x)

        return value, self.action_dist(x), rnn_hxs

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs
