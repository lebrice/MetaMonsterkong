import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.autograd import grad
from math import floor
import random

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, inp):
        return inp

def kld_gauss(mean_1, std_1, mean_2, std_2):
	kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
				(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
				std_2.pow(2) - 1)
	return	0.5 * torch.mean(torch.sum(kld_element, dim=1))

def kld_discrete(prob_1, prob_2):
	return torch.mean(torch.sum(prob_1*(torch.log(prob_1) - torch.log(prob_2)), dim=1))

def nll_discrete(prob, a):
	# Note a is one hot vector
	return -torch.mean(torch.log(torch.sum(prob*a, dim=1)))

def nll_gauss(mean, std, a):
	return torch.mean(torch.sum(((mean - x)/std)**2 + 2*torch.log(std), dim=1))

def sample_normal(mean, std, is_cuda):
	FloatTensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
	eps = FloatTensor(std.size()).normal_()
	return eps.mul(std).add_(mean)

def sample_discrete(prob):
	dist = torch.distributions.Categorical(prob)
	return dist.sample()

def conv2d_size_out(size, kernel_size=1, stride=1, pad=0, dilation=1):
    return floor( ((size + (2 * pad) - ( dilation * (kernel_size - 1) ) - 1 )/ stride) + 1)

