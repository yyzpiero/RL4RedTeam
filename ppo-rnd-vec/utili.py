import numpy as np
import torch
import gym
from stable_baselines3.common.vec_env import VecEnvWrapper





class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        
        actions = actions.cpu().numpy()
        
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems


class Utils():
    def prepro(self, I):
        I = I[35:195]  # crop
        I = I[::2, ::2, 0]  # downsample by factor of 2
        I[I == 144] = 0  # erase background (background type 1)
        I[I == 109] = 0  # erase background (background type 2)
        I[I != 0] = 1  # everything else (paddles, ball) just set to 1
        X = I.astype(np.float32).ravel()  # Combine items in 1 array
        return X

    def count_new_mean(self, prevMean, prevLen, newData):
        return ((prevMean * prevLen) + newData.sum(0)) / (prevLen + newData.shape[0])

    def count_new_std(self, prevStd, prevLen, newData):
        return (((prevStd.pow(2) * prevLen) + (newData.var(0) * newData.shape[0])) / (prevLen + newData.shape[0])).sqrt()

    def normalize(self, data, mean=None, std=None, clip=None):
        if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
            data_normalized = (data - mean) / (std + 1e-8)
        else:
            data_normalized = (data - data.mean()) / (data.std() + 1e-8)

        if clip:
            data_normalized = torch.clamp(data_normalized, -1 * clip, clip)

        return data_normalized

# def make_env(gym_id, seed, idx=0):
#     def thunk():
#         env = gym.make(gym_id)
#         env = gym.wrappers.RecordEpisodeStatistics(env)
#         env.seed(seed)
#         env.action_space.seed(seed)
#         env.observation_space.seed(seed)
#         return env
#     return thunk

def make_env(gym_id, seed, idx=0, wrapper=None):
    def thunk():
        env = gym.make(gym_id)
        if wrapper is not None:
            env = wrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def make_test_env(gym_id, seed, wrapper=None):
    
    env = gym.make(gym_id)
    if wrapper is not None:
        env = wrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env
    #return thunk