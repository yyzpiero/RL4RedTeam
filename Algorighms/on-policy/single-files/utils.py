from gym_minigrid.wrappers import *
import torch
import gym
import nasim

def make_env(env_id, seed, idx):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    def env_thunk():   
    #print(env_id[0:7])
        if env_id[0:8] == "MiniGrid":
            print("=="*10+"MiniGrid"+"=="*10)
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            
            #env = Monitor(env)
            env = OneHotPartialObsWrapper(env)
            #env = RGBImgPartialObsWrapper(env)
            env = FlatObsWrapper(env)
            return env

        if env_id[0:7] == "nasim:c":
            env = nasim.generate(num_hosts=40, num_services=5, num_os=3, num_processes=2, \
                    num_exploits=None, num_privescs=None, r_sensitive=10, r_user=10, \
                    exploit_cost=1, exploit_probs="mixed", privesc_cost=1, privesc_probs=1.0, \
                    service_scan_cost=1, os_scan_cost=1, subnet_scan_cost=1, process_scan_cost=1,\
                    uniform=False, alpha_H=2.0, alpha_V=2.0, lambda_V=1.0, restrictiveness=3, \
                    random_goal=True, base_host_value=1, host_discovery_value=1, \
                    seed=None, name=None, step_limit=10000, address_space_bounds=None)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        
        if env_id[0:7] == "nasim:d":
            env = nasim.generate(num_hosts=200, num_services=5, num_os=3, num_processes=2, \
                    num_exploits=None, num_privescs=None, r_sensitive=10, r_user=10, \
                    exploit_cost=1, exploit_probs="mixed", privesc_cost=1, privesc_probs=1.0, \
                    service_scan_cost=1, os_scan_cost=1, subnet_scan_cost=1, process_scan_cost=1,\
                    uniform=False, alpha_H=2.0, alpha_V=2.0, lambda_V=1.0, restrictiveness=3, \
                    random_goal=True, base_host_value=1, host_discovery_value=1, \
                    seed=None, name=None, step_limit=10000, address_space_bounds=None)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
            
        else:
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            
            return env

    return env_thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def matrix_norm(v, axis=1):
    
    if np.all(v==0):
        return v
    norm = np.divide(v , np.tile(np.sum(v, axis), (v.shape[axis], axis)).transpose())
    
    return norm