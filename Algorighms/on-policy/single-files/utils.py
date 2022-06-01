from gym_minigrid.wrappers import *
import torch
import gym
import copy
import nasim

def make_env_list_random(env_id, seed, num_envs):
    if env_id[0:7] == "nasim:c":
        _fix_env = nasim.generate(num_hosts = 500, 
                    num_os = 3,
                    num_services = 10,
                    num_exploits = 30,
                    num_processes = 3,
                    restrictiveness = 5,
                    step_limit = 500000,
                yz_gen=True, save_fig=True)
    else:
        _fix_env = gym.make(env_id)
    _fix_env = gym.wrappers.RecordEpisodeStatistics(_fix_env)
    _fix_env.seed(seed)
    _fix_env.action_space.seed(seed)
    _fix_env.observation_space.seed(seed)
    
   
        
    def _env_make(fix_env, idx):
        if idx == 0:
            return lambda: fix_env
        else:
            env = copy.deepcopy(fix_env)
            env.seed(seed+idx)
            env.action_space.seed(seed+idx)
            env.observation_space.seed(seed+idx)
            return lambda: env
    
    return [_env_make(_fix_env,i) for i in range(num_envs)]
    

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
            # env = nasim.generate(num_hosts=50, num_services=5, num_os=3, num_processes=2, \
            #         num_exploits=None, num_privescs=None, r_sensitive=10, r_user=10, \
            #         exploit_cost=1, exploit_probs="mixed", privesc_cost=1, privesc_probs=0.9, \
            #         service_scan_cost=1, os_scan_cost=1, subnet_scan_cost=1, process_scan_cost=1,\
            #         uniform=False, alpha_H=2.0, alpha_V=2.0, lambda_V=1.0, restrictiveness=3, \
            #         random_goal=False, base_host_value=1, host_discovery_value=1, \
            #         seed=None, name=None, step_limit=50000, address_space_bounds=None, yz_gen=True, save_fig=True)
            env = nasim.generate(num_hosts = 95, 
                     num_os = 3,
                     num_services = 10,
                     num_exploits = 30,
                     num_processes = 3,
                     restrictiveness = 5,
                     step_limit = 30000,
                    yz_gen=False, save_fig=True)
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