import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from utils import layer_init

class Agent(nn.Module):
    def __init__(self, envs, hidden_size):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, envs.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class RolloutStorage:
    def __init__(self, num_steps, num_envs, observation_space_shape, action_space_shape, device):
        
        self.obs = torch.zeros((num_steps, num_envs) + observation_space_shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + action_space_shape).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)
    
        self.next_done_mask = torch.zeros(num_envs).to(device)

    def return_calculation(self, next_value, use_gae):
        #next_value = agent.get_value(next_obs).reshape(1, -1)
        
        if use_gae:
            self.advantages = torch.zeros_like(self.rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(self.rewards.size(0))):
                if t == self.rewards.size(0) - 1:
                    nextnonterminal = 1.0 - self.next_done_mask
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                _delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = _delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            self.returns = self.advantages + self.values
        else:
            returns = torch.zeros_like(self.rewards).to(device)
            for t in reversed(range(self.rewards.size(0))):
                if t == self.rewards.size(0) - 1:
                    nextnonterminal = 1.0 - self.next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    next_return = returns[t + 1]
                self.returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
            self.advantages = self.returns - self.values
        
        #return returns

    def mini_batch_generator(self, ):


        return 
    