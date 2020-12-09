import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RandomAgent:
    def __init__(self,seed=0,actions=4,observations=10):
        random.seed(seed)
        self.actions = actions
        self.observations = observations

    def learn(self,s,a,r,ns):
        null = 0

    def get_action(self,s):
        return random.randint(0,self.actions-1)


class MLPActorCritic(nn.Module):
    # details from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/networks.py
    def __init__(self, in_size, out_size):
        super(MLPActorCritic, self).__init__()
        #self.in_fn = nn.BatchNorm1d(in_size)
        #self.in_fn.weight.data.fill_(1)
        #self.in_fn.bias.data.fill_(0)
        self.fc1 = nn.Linear(in_size,64)
        self.fc2 = nn.Linear(64,64)
        self.actor_linear = nn.Linear(64, out_size)
        self.critic_linear = nn.Linear(64, out_size)
        
    def forward(self, x):
        #h = self.in_fn(x)
        h = x
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        a = self.actor_linear(h)
        q = self.critic_linear(h)
        return q, a       

class MLPActorCriticValue(nn.Module):
    # details from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/networks.py
    def __init__(self, in_size, out_size):
        super(MLPActorCriticValue, self).__init__()
        #self.in_fn = nn.BatchNorm1d(in_size)
        #self.in_fn.weight.data.fill_(1)
        #self.in_fn.bias.data.fill_(0)
        self.fc1 = nn.Linear(in_size,64)
        self.fc2 = nn.Linear(64,64)
        self.actor_linear = nn.Linear(64, out_size)
        self.critic_linear = nn.Linear(64, 1)
        
    def forward(self, x):
        #h = self.in_fn(x)
        h = x
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        a = self.actor_linear(h)
        v = self.critic_linear(h)
        return v, a       


class CNNActorCriticValue(nn.Module):
    # details from https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/networks.py
    def __init__(self, in_size, out_size):
        super(CNNActorCriticValue, self).__init__()
        print("sizes=",in_size, out_size)
        self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(576, 512)
        self.actor_linear = nn.Linear(512, out_size)
        self.critic_linear = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        logits = self.actor_linear(x)
        value = self.critic_linear(x)
        return value, logits


class PGMLPAgent:
    def __init__(self,seed=0,actions=4,observations=10,lr=0.001,ent=0.1):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actions = actions
        self.observations = observations
        self.network = MLPActorCritic(observations,actions)
        self.network.zero_grad()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = 0.99
        self.ent = ent

    def learn(self,s,a,r,ns):
        action = torch.tensor([[a]])
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        qs, logit = self.network.forward(state_var)
        value = qs.max(-1).values
        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        log_prob = log_prob.gather(1, Variable(action))
        
        # add update here
        qsa = qs.gather(1, Variable(action))
        next_state = np.array([ns])
        next_state_var = Variable(torch.from_numpy(next_state)).float()
        next_qs, next_logit = self.network.forward(next_state_var)
        next_value = next_qs.max(-1).values
        expected_reward = r + self.gamma*next_value
        advantage = expected_reward - value
        td = expected_reward - qsa
        value_loss = 0.5 * td.pow(2)
        policy_loss = -log_prob*Variable(advantage.data) - self.ent*entropy

        (policy_loss + 0.5 * value_loss).backward()
        for param in self.network.parameters():
            param.grad.data.clamp_(-1, 1)
            
        self.optimizer.step()
        self.network.zero_grad()

    def get_action(self,s):
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        qs, logit = self.network.forward(state_var)
        prob = F.softmax(logit)
        action = prob.multinomial(1).data.numpy()[0][0]
        return action

class GAEMLPAgent:
    def __init__(self,seed=0,actions=4,observations=10,lr=0.001,ent=0.1,steps=5):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actions = actions
        self.observations = observations
        self.net = MLPActorCritic(observations,actions)
        self.net.zero_grad()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = 0.99
        self.ent = ent
        self.steps = steps
        self.tau = 0.95
        self.age = 0

        self.state_vars = []
        self.action_vars = []
        self.reward_vars = []
        self.next_state_vars = []
        self.values = []
        self.entropies = []
        self.log_probs = []

    def learn(self,s,a,r,ns):
        self.age += 1
        action = torch.tensor([[a]])
        action_var = Variable(action)
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        reward = np.array([r])
        reward_var = Variable(torch.from_numpy(reward)).float()
        next_state = np.array([ns])
        next_state_var = Variable(torch.from_numpy(next_state)).float()

        self.state_vars.append(state_var)
        self.action_vars.append(action_var)
        self.reward_vars.append(reward_var)
        self.next_state_vars.append(next_state_var)

        qs, logit = self.net.forward(state_var)
        value = qs.max(-1).values
        self.values.append(value)

        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        log_prob = log_prob.gather(1, action_var)
        self.log_probs.append(log_prob)

        if self.age % self.steps == 0:
            gae = torch.zeros(1)
            next_qs, next_logit = self.net.forward(next_state_var)
            next_value = next_qs.max(-1).values
            R = next_value
            self.values.append(next_value)
            policy_loss = 0
            value_loss = 0 
            for i in reversed(range(len(self.reward_vars))):
                R = self.gamma * R + self.reward_vars[i]
                advantage = R - self.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = self.reward_vars[i] + self.gamma*self.values[i + 1].data - self.values[i].data

                gae = gae*self.gamma*self.tau + delta_t

                policy_loss = policy_loss - self.log_probs[i] *Variable(gae) -  self.ent*self.entropies[i]

            ### gradient steps 
            self.net.zero_grad()
              
            (policy_loss.mean() + 0.5 * value_loss.mean()).backward()
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)
                
            self.opt.step()

            self.state_vars = []
            self.action_vars = []
            self.reward_vars = []
            self.next_state_vars = []
            self.values = []
            self.entropies = []
            self.log_probs = []

    def get_action(self,s):
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        qs, logit = self.net.forward(state_var)
        prob = F.softmax(logit)
        action = prob.multinomial(1).data.numpy()[0][0]
        return action

class A2CGAEMLPAgent:
    def __init__(self,seed=0,actions=4,observations=10,lr=0.001,ent=0.1,update_frequency=5):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actions = actions
        self.observations = observations
        self.net = MLPActorCriticValue(observations,actions)
        self.net.zero_grad()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = 0.99
        self.ent = ent
        self.update_frequency = update_frequency
        self.tau = 0.95
        self.age = 0

        self.state_vars = []
        self.action_vars = []
        self.reward_vars = []
        self.next_state_vars = []
        self.values = []
        self.entropies = []
        self.log_probs = []

    def learn(self,s,a,r,ns):
        self.age += 1
        action = torch.tensor([[a]])
        action_var = Variable(action)
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        reward = np.array([r])
        reward_var = Variable(torch.from_numpy(reward)).float()
        next_state = np.array([ns])
        next_state_var = Variable(torch.from_numpy(next_state)).float()

        self.state_vars.append(state_var)
        self.action_vars.append(action_var)
        self.reward_vars.append(reward_var)
        self.next_state_vars.append(next_state_var)

        value, logit = self.net.forward(state_var)
        self.values.append(value)

        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        log_prob = log_prob.gather(1, action_var)
        self.log_probs.append(log_prob)

        if self.age % self.update_frequency == 0:
            gae = torch.zeros(1)
            next_value, next_logit = self.net.forward(next_state_var)
            R = next_value
            self.values.append(next_value)

            policy_loss = 0
            value_loss = 0 
            for i in reversed(range(len(self.reward_vars))):
                R = self.gamma * R + self.reward_vars[i]
                advantage = R - self.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = self.reward_vars[i] + self.gamma*self.values[i + 1].data - self.values[i].data

                gae = gae*self.gamma*self.tau + delta_t

                policy_loss = policy_loss - self.log_probs[i] *Variable(gae) -  self.ent*self.entropies[i]

            ### gradient steps
            self.net.zero_grad()

            (policy_loss.mean() + 0.5 * value_loss.mean()).backward()
            for param in self.net.parameters():
                param.grad.data.clamp_(-1, 1)

            self.opt.step()
            self.state_vars = []
            self.action_vars = []
            self.reward_vars = []
            self.next_state_vars = []
            self.values = []
            self.entropies = []
            self.log_probs = []

    def get_action(self,s):
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        qs, logit = self.net.forward(state_var)
        prob = F.softmax(logit)
        action = prob.multinomial(1).data.numpy()[0][0]
        return action


class A2CGAECNNAgent:
    def __init__(self, seed=0, actions=4, observations=10, lr=0.001, ent=0.1,
                 update_frequency=5, gamma=0.99, log=None, tb_writer=None, args=None):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actions = actions
        self.observations = observations
        self.net = CNNActorCriticValue(observations, actions).to(device)
        self.net.zero_grad()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.ent = ent
        self.update_frequency = update_frequency
        self.tau = 0.95
        self.age = 0

        self.state_vars = []
        self.action_vars = []
        self.reward_vars = []
        self.next_state_vars = []
        self.values = []
        self.entropies = []
        self.log_probs = []

    def learn(self, s, a, r, ns):
        self.age += 1

        action = torch.tensor([[a]])
        action_var = Variable(action).to(device)
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float().to(device)
        state_var = state_var / 255.0 #.permute(0, 3, 1, 2) / 255.  # Channel, Row, Column and normalization
        reward = np.array([r])
        reward_var = Variable(torch.from_numpy(reward)).float().to(device)
        next_state = np.array([ns])
        next_state_var = Variable(torch.from_numpy(next_state)).float().to(device)
        next_state_var = next_state_var / 255.0 #.permute(0, 3, 1, 2) / 255.  # Channel, Row, Column and normalization

        self.state_vars.append(state_var)
        self.action_vars.append(action_var)
        self.reward_vars.append(reward_var)
        self.next_state_vars.append(next_state_var)

        value, logit = self.net.forward(state_var)
        self.values.append(value)

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        log_prob = log_prob.gather(1, action_var)
        self.log_probs.append(log_prob)

        if self.age % self.update_frequency == 0:
            # Compute policy and value loss based on GAE
            gae = torch.zeros(1).to(device)
            next_value, next_logit = self.net.forward(next_state_var)
            R = next_value
            self.values.append(next_value)
            policy_loss = 0
            value_loss = 0
            for i in reversed(range(len(self.reward_vars))):
                R = self.gamma * R + self.reward_vars[i]
                advantage = R - self.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = self.reward_vars[i] + self.gamma * self.values[i + 1].data - self.values[i].data
                gae = gae * self.gamma * self.tau + delta_t
                policy_loss = policy_loss - self.log_probs[i] * Variable(gae) - self.ent * self.entropies[i]

            # Gradient steps
            self.net.zero_grad()
            (policy_loss.mean() + 0.5 * value_loss.mean()).backward()
            for param in self.net.parameters():
                param.grad.data.clamp_(-1., 1.)
            self.opt.step()

            # Re-initialize variables
            self.state_vars = []
            self.action_vars = []
            self.reward_vars = []
            self.next_state_vars = []
            self.values = []
            self.entropies = []
            self.log_probs = []

    def get_action(self,s):
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float().to(device)
        state_var = state_var / 255.0 # .permute(0, 3, 1, 2) / 255.  # Channel, Row, Column and normalization
        qs, logit = self.net.forward(state_var)
        prob = F.softmax(logit, dim=1)
        action = prob.multinomial(1).cpu().data.numpy()[0]
        return action


class MLPLAME(nn.Module):
    def __init__(self,in_size, out_size):
        super(MLPLAME, self).__init__()
        self.fc1 = nn.Linear(in_size,64)
        self.fc2 = nn.Linear(64,64)
        self.actor_linear = nn.Linear(64, out_size)
        self.critic_linear = nn.Linear(64, out_size)
        
    def forward(self, x, weights=None):
        if weights == None:
            h1 = F.relu(self.fc1(x))
            h2 = F.relu(self.fc2(h1))
            a = self.actor_linear(h2)
            q = self.critic_linear(h2)
        else:
            h1 = F.relu(nn.functional.linear(x, weights[0], weights[1]))
            h2 = F.relu(nn.functional.linear(h1, weights[2], weights[3]))
            a = nn.functional.linear(h2, weights[4], weights[5])
            q = nn.functional.linear(h2, weights[6], weights[7])
        return q, a

class MetaGAEMLPAgent:
    def __init__(self,seed=0,actions=4,target=False,observations=10,lr=0.001,alpha=0.001,beta=0.001,ent=0.1,steps=5):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actions = actions
        self.observations = observations
        self.net = MLPLAME(observations,actions)
        self.net.zero_grad()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = 0.99
        self.ent = ent
        self.steps = steps
        self.inner_lr = alpha
        self.tau = 0.95
        self.age = 0

        self.state_vars = []
        self.action_vars = []
        self.reward_vars = []
        self.next_state_vars = []
        self.values = []
        self.entropies = []
        self.log_probs = []

        self.weights = list(self.net.parameters())
        self.temp_weights = {}
        self.tmp_idx = 0
        self.temp_weights[self.tmp_idx] = [w.clone() for w in self.weights]
        self.target = target
        self.beta = beta

    def learn(self,s,a,r,ns):
        self.age += 1
        action = torch.tensor([[a]])
        action_var = Variable(action)
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        reward = np.array([r])
        reward_var = Variable(torch.from_numpy(reward)).float()
        next_state = np.array([ns])
        next_state_var = Variable(torch.from_numpy(next_state)).float()

        self.state_vars.append(state_var)
        self.action_vars.append(action_var)
        self.reward_vars.append(reward_var)
        self.next_state_vars.append(next_state_var)
        self.net.zero_grad()
        
        qs, logit = self.net.forward(state_var,weights=self.temp_weights[self.tmp_idx])
        value = qs.max(-1).values
        self.values.append(value)

        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        log_prob = log_prob.gather(1, action_var)
        self.log_probs.append(log_prob)

        # inner loop
        qsa = qs.gather(1, action_var)
        next_qs, next_logit = self.net.forward(next_state_var,weights=self.temp_weights[self.tmp_idx])
        next_value = next_qs.max(-1).values
        expected_reward = reward_var + self.gamma*next_value
        advantage = expected_reward - value

        td = expected_reward - qsa
        value_loss = 0.5 * td.pow(2)
        policy_loss = -log_prob*Variable(advantage.data) - self.ent*entropy

        loss = policy_loss.mean() + 0.5 * value_loss.mean()

        grad = torch.autograd.grad(loss, self.temp_weights[self.tmp_idx], create_graph=True)
        if self.tmp_idx == 0:
            self.first_grad = grad
            
        self.temp_weights[self.tmp_idx+1] = []
        for w,g in zip(self.temp_weights[self.tmp_idx], grad):
            self.temp_weights[self.tmp_idx+1].append(w - self.inner_lr * g.data.clamp_(-1, 1))

        self.tmp_idx += 1

        if self.age % self.steps == 0:
            gae = torch.zeros(1)
            next_qs, next_logit = self.net.forward(next_state_var,weights=self.temp_weights[self.tmp_idx])
            next_value = next_qs.max(-1).values
            R = next_value
            self.values.append(next_value)
            policy_loss = 0
            value_loss = 0 
            for i in reversed(range(len(self.reward_vars))):
                R = self.gamma * R + self.reward_vars[i]
                advantage = R - self.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = self.reward_vars[i] + self.gamma*self.values[i + 1].data - self.values[i].data

                gae = gae*self.gamma*self.tau + delta_t

                policy_loss = policy_loss - self.log_probs[i] *Variable(gae) -  self.ent*self.entropies[i]
              
            ### meta-gradient steps
            self.net.zero_grad()
            meta_loss = policy_loss.mean() + 0.5 * value_loss.mean()

            meta_grads = torch.autograd.grad(meta_loss, self.weights)
            for w, g in zip(self.weights, meta_grads):
                w.grad = g.data.clamp_(-1, 1)

            # assign meta gradient to weights and take optimisation step
            #if self.target:
            #    meta_grads = torch.autograd.grad(meta_loss, self.weights, create_graph=True)
            #    align_loss = 0.0
            #    denom = 0.0
            #    for g, m in zip(self.first_grad, meta_grads):
            #        align_loss += torch.mean((g.view(-1)-m.view(-1))**2)                   
            #        denom += 1.0
                    
            #    align_loss = torch.clamp(self.beta*align_loss/denom,max=100.0,min=-100.0) # factor in beta weighting and clipping mse in range
            #    align_grads = torch.autograd.grad(align_loss, self.weights)

            #    for w, g, a in zip(self.weights, meta_grads, align_grads):
            #        w.grad = g.data.clamp_(-1, 1) + a.data.clamp_(-1, 1)

            #else:
            #    meta_grads = torch.autograd.grad(meta_loss, self.weights, create_graph=True)
            #    align_loss = 0.0
            #    denom = 0.0
            #    for g, m in zip(self.first_grad, meta_grads):
            #        align_loss += torch.mean((g.view(-1)-Variable(m.data.view(-1)))**2)                  
            #        denom += 1.0
                    
            #    align_loss = torch.clamp(self.beta*align_loss/denom,max=100.0,min=-100.0) # factor in beta weighting and clipping mse in range
            #    align_grads = torch.autograd.grad(align_loss, self.weights)

            #    for w, g, a in zip(self.weights, meta_grads, align_grads):
            #        w.grad = g.data.clamp_(-1, 1) + a.data.clamp_(-1, 1)
            
            self.opt.step()

            self.state_vars = []
            self.action_vars = []
            self.reward_vars = []
            self.next_state_vars = []
            self.values = []
            self.entropies = []
            self.log_probs = []

            self.weights = list(self.net.parameters())
            self.temp_weights = {}
            self.tmp_idx = 0
            self.temp_weights[self.tmp_idx] = [w.clone() for w in self.weights]

    def get_action(self,s):
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        qs, logit = self.net.forward(state_var)
        prob = F.softmax(logit)
        action = prob.multinomial(1).data.numpy()[0][0]
        return action


class LAMEMLPAgent:
    def __init__(self,seed=0,actions=4,observations=10,lr=0.001,ilr=0.001,ent=0.1):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actions = actions
        self.observations = observations
        self.network = MLPLAME(observations,actions)
        self.network.zero_grad()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = 0.99
        self.ent = ent
        self.ilr = ilr

    def learn(self,s,a,r,ns):
        action = torch.tensor([[a]])
#        state = np.array([s])
#        state_var = Variable(torch.from_numpy(state)).float()
#        qs, logit = self.network.forward(state_var)
#        value = qs.max(-1).values
#        prob = F.softmax(logit)
#        log_prob = F.log_softmax(logit)
#        log_prob = log_prob.gather(1, Variable(action))
#        entropy = -(log_prob * prob).sum(1)
        
        # add update here
#        qsa = qs.gather(1, Variable(action))
#        next_state = np.array([ns])
#        next_state_var = Variable(torch.from_numpy(next_state)).float()
#        next_qs, next_logit = self.network.forward(next_state_var)
#        next_value = next_qs.max(-1).values
#        expected_reward = r + self.gamma*next_value
#        advantage = expected_reward - value
#        td = expected_reward - qsa
#        value_loss = 0.5 * td.pow(2)
#        policy_loss = -log_prob*Variable(advantage.data) - self.ent*entropy

#        (policy_loss + 0.5 * value_loss).backward()
#        for param in self.network.parameters():
#            param.grad.data.clamp_(-1, 1)
            
#        self.optimizer.step()
#        self.network.zero_grad()
        weights = list(self.network.parameters())
        temp_weights = {}
        temp_weights[0] = [w.clone() for w in weights]
        
        ### inner loop
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        qs, logit = self.network.forward(state_var,weights=temp_weights[0])
        value = qs.max(-1).values
        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        log_prob = log_prob.gather(1, Variable(action))
        
        
        qsa = qs.gather(1, Variable(action))
        next_state = np.array([ns])
        next_state_var = Variable(torch.from_numpy(next_state)).float()
        next_qs, next_logit = self.network.forward(next_state_var,weights=temp_weights[0])
        next_value = next_qs.max(-1).values
        expected_reward = r + self.gamma*next_value
        advantage = expected_reward - value
        td = expected_reward - qsa
        value_loss = 0.5 * td.pow(2)
        policy_loss = -log_prob*Variable(advantage.data) - self.ent*entropy

        loss = policy_loss.mean() + 0.5 * value_loss.mean()
        grad = torch.autograd.grad(loss, temp_weights[0], create_graph=True)
        temp_weights[1] = []
        for w,g in zip(temp_weights[0], grad):
            temp_weights[1].append(w - self.ilr * g.data.clamp_(-1, 1))


        ### outer loop
        # --- WRONG? qs, logit = self.network.forward(state_var,weights=temp_weights[1])
        # --- WRONG? value = qs.max(-1).values.view(-1,1)
        next_state_var = Variable(torch.from_numpy(next_state)).float()
        next_qs, next_logit = self.network.forward(next_state_var,weights=temp_weights[1])
        next_value = next_qs.max(-1).values.view(-1,1)
        expected_reward = r + self.gamma*next_value
        advantage = expected_reward - value

        td = expected_reward - qsa
        value_loss = 0.5 * td.pow(2)
        policy_loss = -log_prob*Variable(advantage.data) - self.ent*entropy.view(-1,1)

        meta_loss = policy_loss.mean() + 0.5 * value_loss.mean()

        meta_grads = torch.autograd.grad(meta_loss, weights)
        # assign meta gradient to weights and take optimisation step
        for w, g in zip(weights, meta_grads):
            w.grad = g.data.clamp_(-1, 1)

        self.optimizer.step()
        self.network.zero_grad()

        

    def get_action(self,s):
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        qs, logit = self.network.forward(state_var)
        prob = F.softmax(logit)
        action = prob.multinomial(1).data.numpy()[0][0]
        return action



class MLPRL2L(nn.Module):
    def __init__(self,in_size, out_size):
        super(MLPRL2L, self).__init__()
        self.fc1 = nn.Linear(in_size,64)
        self.fc2 = nn.Linear(64,64)
        self.actor_linear = nn.Linear(64, out_size)
        self.critic_linear = nn.Linear(64, 1)
        
    def forward(self, x, weights=None):
        if weights == None:
            h1 = F.relu(self.fc1(x))
            h2 = F.relu(self.fc2(h1))
            a = self.actor_linear(h2)
            v = self.critic_linear(h2)
        else:
            h1 = F.relu(nn.functional.linear(x, weights[0], weights[1]))
            h2 = F.relu(nn.functional.linear(h1, weights[2], weights[3]))
            a = nn.functional.linear(h2, weights[4], weights[5])
            v = nn.functional.linear(h2, weights[6], weights[7])
        return v, a


class CNNRL2L(nn.Module):
    def __init__(self,in_size, out_size):
        super(CNNRL2L, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(576, 512)
        self.actor_linear = nn.Linear(512, out_size)
        self.critic_linear = nn.Linear(512, 1)

    def forward(self, x, weights=None):
        if weights == None:
            x = F.relu(self.maxp1(self.conv1(x)))
            x = F.relu(self.maxp2(self.conv2(x)))
            x = F.relu(self.maxp3(self.conv3(x)))
            x = F.relu(self.maxp4(self.conv4(x)))
            x = torch.flatten(x, start_dim=1)
            x = F.relu(self.fc1(x))
            logits = self.actor_linear(x)
            value = self.critic_linear(x)
        else:
            x = F.relu(F.max_pool2d(F.conv2d(x, weight=weights[0], bias=weights[1], stride=1, padding=2), 2))
            x = F.relu(F.max_pool2d(F.conv2d(x, weight=weights[2], bias=weights[3], stride=1, padding=1), 2))
            x = F.relu(F.max_pool2d(F.conv2d(x, weight=weights[4], bias=weights[5], stride=1, padding=1), 2))
            x = F.relu(F.max_pool2d(F.conv2d(x, weight=weights[6], bias=weights[7], stride=1, padding=1), 2))
            x = torch.flatten(x, start_dim=1)
            x = F.relu(F.linear(x, weight=weights[8], bias=weights[9]))
            logits = F.linear(x, weight=weights[10], bias=weights[11])
            value = F.linear(x, weight=weights[12], bias=weights[13])
        return value, logits

class RL2LGAEMLPAgent:
    def __init__(self,seed=0,actions=4,target=False,observations=10,lr=0.001,alpha=0.001,beta=0.0,ent=0.1,update_frequency=5):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actions = actions
        self.observations = observations
        self.net = MLPRL2L(observations,actions)
        self.net.zero_grad()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = 0.99
        self.ent = ent
        self.update_frequency = update_frequency
        self.inner_lr = alpha
        self.tau = 0.95
        self.age = 0

        self.state_vars = []
        self.action_vars = []
        self.reward_vars = []
        self.next_state_vars = []
        self.values = []
        self.entropies = []
        self.log_probs = []

        self.weights = list(self.net.parameters())
        self.temp_weights = {}
        self.tmp_idx = 0
        self.temp_weights[self.tmp_idx] = [w.clone() for w in self.weights]
        self.target = target
        self.beta = beta # if beta=0, then do not apply penalty semi-gradient

    def learn(self,s,a,r,ns):
        self.age += 1
        action = torch.tensor([[a]])
        action_var = Variable(action)
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        reward = np.array([r])
        reward_var = Variable(torch.from_numpy(reward)).float()
        next_state = np.array([ns])
        next_state_var = Variable(torch.from_numpy(next_state)).float()

        self.state_vars.append(state_var)
        self.action_vars.append(action_var)
        self.reward_vars.append(reward_var)
        self.next_state_vars.append(next_state_var)
        self.net.zero_grad()
        
        value, logit = self.net.forward(state_var,weights=self.temp_weights[self.tmp_idx])
        self.values.append(value)

        prob = F.softmax(logit)
        log_prob = F.log_softmax(logit)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        log_prob = log_prob.gather(1, action_var)
        self.log_probs.append(log_prob)

        # inner loop
        next_value, next_logit = self.net.forward(next_state_var,weights=self.temp_weights[self.tmp_idx])
        expected_reward = reward_var + self.gamma*next_value
        advantage = expected_reward - value

        td = expected_reward - value
        value_loss = 0.5 * td.pow(2)
        policy_loss = -log_prob*Variable(advantage.data) - self.ent*entropy

        loss = policy_loss.mean() + 0.5 * value_loss.mean()

        grad = torch.autograd.grad(loss, self.temp_weights[self.tmp_idx], create_graph=True)
        if self.tmp_idx == 0:
            self.first_grad = grad
            
        self.temp_weights[self.tmp_idx+1] = []
        for w,g in zip(self.temp_weights[self.tmp_idx], grad):
            self.temp_weights[self.tmp_idx+1].append(w - self.inner_lr * g.data.clamp_(-1, 1))

        self.tmp_idx += 1

        if self.age % self.update_frequency == 0:
            gae = torch.zeros(1)
            next_value, next_logit = self.net.forward(next_state_var,weights=self.temp_weights[self.tmp_idx])
            R = next_value
            self.values.append(next_value)
            policy_loss = 0
            value_loss = 0 
            for i in reversed(range(len(self.reward_vars))):
                R = self.gamma * R + self.reward_vars[i]
                advantage = R - self.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = self.reward_vars[i] + self.gamma*self.values[i + 1].data - self.values[i].data

                gae = gae*self.gamma*self.tau + delta_t

                policy_loss = policy_loss - self.log_probs[i] *Variable(gae) -  self.ent*self.entropies[i]
              
            ### meta-gradient steps
            self.net.zero_grad()
            meta_loss = policy_loss.mean() + 0.5 * value_loss.mean()

            if self.beta == 0.0:
                # run without penalty semi-gradient
                meta_grads = torch.autograd.grad(meta_loss, self.weights)
                for w, g in zip(self.weights, meta_grads):
                    w.grad = g.data.clamp_(-1, 1)
            else:
                #compute beta weighted penalty semi-gradient
                meta_grads = torch.autograd.grad(meta_loss, self.weights, create_graph=True)
                align_loss = 0.0
                denom = 0.0
                for g, m in zip(self.first_grad, meta_grads):
                    align_loss += torch.mean((g.view(-1)-Variable(m.data.view(-1)))**2)                  
                    denom += 1.0
                    
                align_loss = torch.clamp(self.beta*align_loss/denom,max=100.0,min=-100.0) # factor in beta weighting and clipping mse in range
                align_grads = torch.autograd.grad(align_loss, self.weights)
                for w, g, a in zip(self.weights, meta_grads, align_grads):
                    w.grad = g.data.clamp_(-1, 1) + a.data.clamp_(-1, 1)
            
            self.opt.step()

            self.state_vars = []
            self.action_vars = []
            self.reward_vars = []
            self.next_state_vars = []
            self.values = []
            self.entropies = []
            self.log_probs = []

            self.weights = list(self.net.parameters())
            self.temp_weights = {}
            self.tmp_idx = 0
            self.temp_weights[self.tmp_idx] = [w.clone() for w in self.weights]

    def get_action(self,s):
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        qs, logit = self.net.forward(state_var)
        prob = F.softmax(logit)
        action = prob.multinomial(1).data.numpy()[0][0]
        return action


class RL2LGAECNNAgent:
    def __init__(self, seed=0, actions=4, target=False, observations=10, lr=0.001,
                alpha=0.001, beta=0.0, ent=0.1, update_frequency=5, gamma=0.99,
                log=None, tb_writer=None, args=None):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.actions = actions
        self.observations = observations
        self.net = CNNRL2L(observations, actions)
        self.net.zero_grad()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.ent = ent
        self.update_frequency = update_frequency
        self.inner_lr = alpha
        self.tau = 0.95
        self.age = 0

        self.state_vars = []
        self.action_vars = []
        self.reward_vars = []
        self.next_state_vars = []
        self.values = []
        self.entropies = []
        self.log_probs = []

        self.weights = list(self.net.parameters())
        self.temp_weights = {}
        self.tmp_idx = 0
        self.temp_weights[self.tmp_idx] = [w.clone() for w in self.weights]
        self.target = target
        self.beta = beta # if beta=0, then do not apply penalty semi-gradient

    def learn(self, s, a, r, ns):
        self.age += 1
        action = torch.tensor([[a]])
        action_var = Variable(action)
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        state_var = state_var / 255.0 #.permute(0, 3, 1, 2) / 255.  # Channel, Row, Column and normalization
        reward = np.array([r])
        reward_var = Variable(torch.from_numpy(reward)).float()
        next_state = np.array([ns])
        next_state_var = Variable(torch.from_numpy(next_state)).float()
        next_state_var = next_state_var / 255.0 #.permute(0, 3, 1, 2) / 255.  # Channel, Row, Column and normalization

        self.state_vars.append(state_var)
        self.action_vars.append(action_var)
        self.reward_vars.append(reward_var)
        self.next_state_vars.append(next_state_var)
        self.net.zero_grad()

        value, logit = self.net.forward(state_var, weights=self.temp_weights[self.tmp_idx])
        self.values.append(value)

        prob = F.softmax(logit, dim=1)
        log_prob = F.log_softmax(logit, dim=1)
        entropy = -(log_prob * prob).sum(1)
        self.entropies.append(entropy)
        log_prob = log_prob.gather(1, action_var)
        self.log_probs.append(log_prob)

        # Inner update
        next_value, next_logit = self.net.forward(next_state_var, weights=self.temp_weights[self.tmp_idx])
        expected_reward = reward_var + self.gamma * next_value
        advantage = expected_reward - value

        td = expected_reward - value
        value_loss = 0.5 * td.pow(2)
        policy_loss = -log_prob * Variable(advantage.data) - self.ent * entropy
        loss = policy_loss.mean() + 0.5 * value_loss.mean()

        grad = torch.autograd.grad(loss, self.temp_weights[self.tmp_idx], create_graph=True)
        if self.tmp_idx == 0:
            self.first_grad = grad

        self.temp_weights[self.tmp_idx + 1] = []
        for w, g in zip(self.temp_weights[self.tmp_idx], grad):
            self.temp_weights[self.tmp_idx + 1].append(w - self.inner_lr * g.data.clamp_(-1, 1))

        self.tmp_idx += 1

        if self.age % self.update_frequency == 0:
            gae = torch.zeros(1)
            next_value, next_logit = self.net.forward(next_state_var, weights=self.temp_weights[self.tmp_idx])
            R = next_value
            self.values.append(next_value)
            policy_loss = 0
            value_loss = 0
            for i in reversed(range(len(self.reward_vars))):
                R = self.gamma * R + self.reward_vars[i]
                advantage = R - self.values[i]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = self.reward_vars[i] + self.gamma*self.values[i + 1].data - self.values[i].data
                gae = gae * self.gamma * self.tau + delta_t
                policy_loss = policy_loss - self.log_probs[i] * Variable(gae) - self.ent * self.entropies[i]

            # Meta-gradient steps
            self.net.zero_grad()
            meta_loss = policy_loss.mean() + 0.5 * value_loss.mean()

            if self.beta == 0.0:
                # run without penalty semi-gradient
                meta_grads = torch.autograd.grad(meta_loss, self.weights)
                for w, g in zip(self.weights, meta_grads):
                    w.grad = g.data.clamp_(-1, 1)
            else:
                #compute beta weighted penalty semi-gradient
                meta_grads = torch.autograd.grad(meta_loss, self.weights, create_graph=True)
                align_loss = 0.0
                denom = 0.0
                for g, m in zip(self.first_grad, meta_grads):
                    align_loss += torch.mean((g.view(-1)-Variable(m.data.view(-1)))**2)
                    denom += 1.0

                align_loss = torch.clamp(self.beta*align_loss/denom,max=100.0,min=-100.0) # factor in beta weighting and clipping mse in range
                align_grads = torch.autograd.grad(align_loss, self.weights)
                for w, g, a in zip(self.weights, meta_grads, align_grads):
                    w.grad = g.data.clamp_(-1, 1) + a.data.clamp_(-1, 1)

            self.opt.step()

            self.state_vars = []
            self.action_vars = []
            self.reward_vars = []
            self.next_state_vars = []
            self.values = []
            self.entropies = []
            self.log_probs = []

            self.weights = list(self.net.parameters())
            self.temp_weights = {}
            self.tmp_idx = 0
            self.temp_weights[self.tmp_idx] = [w.clone() for w in self.weights]

    def get_action(self,s):
        state = np.array([s])
        state_var = Variable(torch.from_numpy(state)).float()
        state_var = state_var / 255.0 #.permute(0, 3, 1, 2) / 255.  # Channel, Row, Column and normalization
        qs, logit = self.net.forward(state_var)
        prob = F.softmax(logit, dim=1)
        action = prob.multinomial(1).data.numpy()[0]
        return action
