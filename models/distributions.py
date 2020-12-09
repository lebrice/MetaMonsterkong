import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.utils import AddBias, init
import gym
import numpy as np

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

class FixedTanhNormal(torch.distributions.Distribution):
    """
    Represent distribution of X where
        X ~ tanh(Z)
        Z ~ N(mean, std)
    Note: this is not very numerically stable.
    """
    def __init__(self, normal_mean, normal_std, device, epsilon=1e-6):
        """
        :param normal_mean: Mean of the normal distribution
        :param normal_std: Std of the normal distribution
        :param epsilon: Numerical stability epsilon when computing log-prob.
        """
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.normal = torch.distributions.Normal(normal_mean, normal_std)
        self.epsilon = epsilon
        self.device = device

    def sample_n(self, n, return_pre_tanh_value=False):
        z = self.normal.sample_n(n)
        if return_pre_tanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def log_prob(self, value, pre_tanh_value=None):
        """
        :param value: some value, x
        :param pre_tanh_value: arctanh(x)
        :return:
        """
        if pre_tanh_value is None:
            pre_tanh_value = torch.log(
                (1+value) / (1-value)
            ) / 2
        return self.normal.log_prob(pre_tanh_value) - torch.log(
            1 - value * value + self.epsilon
        )

    def mode(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """        
        z = self.normal_mean

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def sample(self, return_pretanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        See https://github.com/pytorch/pytorch/issues/4620 for discussion.
        """
        z = self.normal.sample().detach()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)

    def rsample(self, return_pretanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = (
            self.normal_mean +
            self.normal_std *
            torch.distributions.Normal(
                torch.zeros(self.normal_mean.size()).to(self.device),
                torch.ones(self.normal_std.size()).to(self.device)
            ).sample()
        )
        z.requires_grad_()

        if return_pretanh_value:
            return torch.tanh(z), z
        else:
            return torch.tanh(z)        

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        sample = super(FixedCategorical, self).sample()
        return sample.unsqueeze(-1)

    def log_probs(self, actions):
        log_probs = super(FixedCategorical, self).log_prob(actions.squeeze(-1))
        return log_probs.view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def get_probs(self):
        return self.probs
    


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        log_probs = super(FixedNormal, self).log_prob(actions)
        return log_probs.sum(-1, keepdim=True)

    def mode(self):
        return self.mean

    def entropy(self):
        entropy = super(FixedNormal, self).entropy()
        return entropy.sum(-1)

    def get_probs(self):
        return self.mean
    

# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        log_probs = super(FixedBernoulli, self).log_prob(actions)
        return log_probs.view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()

    def entropy(self):
        entropy = super(FixedBernoulli, self).entropy()
        return entropy.sum(-1)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))
        self.space = gym.spaces.Discrete(num_outputs)

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
    


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_std=1.0):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.log(init_std*torch.ones(num_outputs)))
        self.space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_outputs,))

    def forward(self, x, res=None):
        action_mean = self.fc_mean(x)

        if res is not None:
            action_mean += res
            
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())

class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)


class TanhDiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, device, init_w_m=3e-3, init_w_ls=1e-3, clamp={'LOG_SIG_MAX':2, 'LOG_SIG_MIN':-20}):
        super(TanhDiagGaussian, self).__init__()
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.fc_mean.weight.data.uniform_(-init_w_m, init_w_m)
        self.fc_mean.bias.data.uniform_(-init_w_m, init_w_m)
        
        self.log_std = nn.Linear(num_inputs, num_outputs)
        self.log_std.weight.data.uniform_(-init_w_ls, init_w_ls)
        self.log_std.bias.data.uniform_(-init_w_ls, init_w_ls)

        self.space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(num_outputs,))
        self.clamp = clamp
        self.device = device

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_logstd = self.log_std(x)
        action_logstd = torch.clamp(action_logstd, self.clamp['LOG_SIG_MIN'], self.clamp['LOG_SIG_MAX'])
        return FixedTanhNormal(action_mean, action_logstd.exp(), device=self.device)    