from turtle import done
import gym
import os 
import sys
import numpy as np
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
import nasim
# from utils.rl_tools import SaveOnBestTrainingRewardCallback, env_create, eval_agent
# from gym_minigrid.wrappers import *

since = time.time()
# env = RGBImgPartialObsWrapper(env) # Get pixel observations
# env = ImgObsWrapper(env) # Get rid of the 'mission' field
# obs = env.reset() # This now produces an RGB tensor only

# log_dir = "./tmp/gym/"
# log_dir = os.path.abspath(log_dir)
# print(log_dir)
# os.makedirs(log_dir, exist_ok=True)

# Create environment
env = gym.make("nasim:Small-v0")
# env = DummyVecEnv([lambda:env])

# env = nasim.generate(num_hosts=26, num_services=5, num_os=3, num_processes=2, \
#                     num_exploits=None, num_privescs=None, r_sensitive=10, r_user=10, \
#                     exploit_cost=1, exploit_probs="mixed", privesc_cost=1, privesc_probs=1.0, \
#                     service_scan_cost=1, os_scan_cost=1, subnet_scan_cost=1, process_scan_cost=1,\
#                     uniform=False, alpha_H=2.0, alpha_V=2.0, lambda_V=1.0, restrictiveness=3, \
#                     random_goal=True, base_host_value=1, host_discovery_value=1, \
#                     seed=None, name=None, step_limit=10000, address_space_bounds=None, yz_gen=True)
env = gym.wrappers.RecordEpisodeStatistics(env)
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
#env = gym.make('nasim:Small-v0')
# test_env = gym.make('nasim:Small-PO-v0')
# env = gym.make('MiniGrid-DoorKey-5x5-v0')
# env = RGBImgPartialObsWrapper(env)
# env = FlatObsWrapper(env)
#env.reset()
# env = make_vec_env("nasim:Small-PO-v0", n_envs=1, monitor_dir=log_dir)
#env = env_create(env_id="nasim:Medium-v0")
#env = Monitor(gym.make("nasim:Tiny-PO-v0"),allow_early_resets=False)
# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)
model.gamma = 0.9
#callback = SaveOnBestTrainingRewardCallback(check_freq=20, log_dir=log_dir, verbose=0, idx=0)

# Train the agent
model.learn(total_timesteps=int(100000))#, callback=callback)

#params = model.get_parameters()

# Save the agent
#model.save("dqn_lunar")
#del model  # delete trained model to demonstrate loading

# Load the trained agent
# NOTE: if you have loading issue, you can pass `print_system_info=True`
# to compare the system on which the model was trained vs the current one
# model = DQN.load("dqn_lunar", env=env, print_system_info=True)
#model = PPO.load(log_dir+"/best_model/"+"ID_0_Best_Model", env=env)

#model.set_parameters(params) 
#model.gamma = 0.8
#model.learn(total_timesteps=int(10000))#, callback=callback)
# Evaluate the agent
# NOTE: If you use wrappers with your environment that modify rewards,
#       this will be reflected here. To evaluate with original rewards,
#       wrap environment in a "Monitor" wrapper before other wrappers.
#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2)
#mean_reward= eval_agent(model, env)
#print(mean_reward)

# # Enjoy trained agent
# obs = env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, rewards, dones, info = env.step(action)
#     #env.render()


time_elapsed = time.time()-since
print("Total Run Time: {}".format(time_elapsed))