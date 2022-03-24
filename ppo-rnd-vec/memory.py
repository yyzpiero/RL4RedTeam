from re import S
import torch
import numpy as np
from torch.utils.data import Dataset
from utili import Utils

class RolloutRunner():
    def __init__(self, state_dim, device='cuda', rnd=True):
        self.rollout_buffer = Memory()
        if rnd:
            self.obs_buffer = ObsMemory(state_dim, device)
        self.device=device
        self.utils = Utils()

    def save_eps(self, state, action, reward, done, next_state, coverage_hist=1):
        self.rollout_buffer.save_eps(state, action, reward, done, next_state, coverage_hist)

    def save_observation(self, obs):
       
        self.obs_buffer.save_eps(obs)
    
    def update_obs_normalization_param(self, obs):
          
        obs = torch.FloatTensor(obs).to(self.device).detach()

        mean_obs = self.utils.count_new_mean(self.obs_buffer.mean_obs, self.obs_buffer.total_number_obs, obs)
        std_obs = self.utils.count_new_std(self.obs_buffer.std_obs, self.obs_buffer.total_number_obs, obs)
        total_number_obs = len(obs) + self.obs_buffer.total_number_obs

        self.obs_buffer.save_observation_normalize_parameter(mean_obs, std_obs, total_number_obs)

    def update_rwd_normalization_param(self, in_rewards):
        std_in_rewards = self.utils.count_new_std(
            self.obs_buffer.std_in_rewards, self.obs_buffer.total_number_rwd, in_rewards)
        total_number_rwd = len(in_rewards) + self.obs_buffer.total_number_rwd

        self.obs_buffer.save_rewards_normalize_parameter(
            std_in_rewards, total_number_rwd)




class ObsMemory(Dataset):
    def __init__(self, state_dim, device="cuda"):
        self.observations = []

        self.mean_obs = torch.zeros(state_dim).to(device)
        self.std_obs = torch.zeros(state_dim).to(device)
        self.std_in_rewards = torch.zeros(1).to(device)
        self.total_number_obs = torch.zeros(1).to(device)
        self.total_number_rwd = torch.zeros(1).to(device)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return np.array(self.observations[idx], dtype=np.float32)

    def get_all(self):
        return torch.FloatTensor(np.array(self.observations))

    def save_eps(self, obs):
        self.observations.append(obs)

    def save_observation_normalize_parameter(self, mean_obs, std_obs, total_number_obs):
        self.mean_obs = mean_obs
        self.std_obs = std_obs
        self.total_number_obs = total_number_obs

    def save_rewards_normalize_parameter(self, std_in_rewards, total_number_rwd):
        self.std_in_rewards = std_in_rewards
        self.total_number_rwd = total_number_rwd

    def clear_memory(self):
        del self.observations[:]


class Memory(Dataset):
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.coverage_hist = []

    def __len__(self):
        return len(self.dones)

    def __getitem__(self, idx):
        return np.array(self.states[idx], dtype=np.float32), np.array(self.actions[idx], dtype=np.float32), np.array([self.rewards[idx]], dtype=np.float32), np.array([self.dones[idx]], dtype=np.float32), np.array(self.next_states[idx], dtype=np.float32), np.array(self.coverage_hist[idx], dtype=np.float32)
        #return self.states[idx], self.actions[idx], self.rewards[idx], self.dones[idx], self.next_states[idx], self.coverage_hist[idx]

    def fetch_all(self, numpy=True, to_device="cpu"):
        if numpy:
            return np.array(self.states, dtype=np.float32), np.array(self.actions, dtype=np.float32), np.array(self.rewards, dtype=np.float32), np.array(self.dones, dtype=np.float32), np.array(self.next_states, dtype=np.float32), np.array(self.coverage_hist, dtype=np.float32)
        else:
            return torch.FloatTensor(np.array(self.states)).to(to_device), torch.FloatTensor(np.array(self.actions)).to(to_device), \
                      torch.FloatTensor(self.rewards).to(to_device), torch.FloatTensor(self.dones).to(to_device), torch.FloatTensor(self.next_states).to(to_device),\
                      torch.FloatTensor(self.coverage_hist).to(to_device)
    

    def save_eps(self, state, action, reward, done, next_state, coverage_hist=324):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.coverage_hist.append(coverage_hist)

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]
        del self.coverage_hist[:]