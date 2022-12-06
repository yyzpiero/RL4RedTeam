import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from tensorboardX import SummaryWriter

from agent import PPO_Agent
from distribution import * 
from utili import *
from networks import *
from memory import RolloutRunner
from eps_control import run_inits_steps, rollout_steps, test_episode

# Uncomment this if we'd like to use CybORG
#from CybORG import CybORG
#from CybORG.Agents.Wrappers import *
#from gym_minigrid.wrappers import *

# Main Hyper Parameteres
PPO_HYPERPARAMS = {

    "n_plot_batch": 10000000,  # How many episode you want to plot the result
    "n_episode":  10000,  # How many episode you want to run
    "n_init_episode": 1024,  # default 1014
    #n_init_episode = 32
    "n_saved":10,  # How many episode to run before saving the weights
    "num_hidden": 512,
    "anneal_lr": False,

    "policy_kl_range": 0.0008,  # Recommended set to 0.0008 for Discrete
    "policy_params":  20,  # Recommended set to 20 for Discrete
    "value_clip": 10.0,  # How many value will be clipped. Recommended set to the highest or lowest possible reward
    "entropy_coef": 0.025, # How much randomness of action you will get
    "vf_loss_coef": 1,  # Just set to 1
    "minibatch": 4,  # How many batch per update. size of batch = n_update / minibatch. Recommended set to 4 for Discrete
    "PPO_epochs": 5,  # How many epoch per update. Recommended set to 10 for Discrete

    "gamma": 0.99,  # Just set to 0.99
    "lam": 0.95,  # Just set to 0.95
    "learning_rate": 2.5e-4,  # Just set to 0.95
    
    "mask_type": None, # "Soft" mask actor logits, "Hard" mask just cut-off the softmax output 
    "coverage": False,
    "coverage_coef": 0.15,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


def main():
    ############## Hyperparameters ##############
    load_weights = False  # If you want to load the agent, set this to True
    save_weights = False  # If you want to save the agent, set this to True
    # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    training_mode = True
    # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    reward_threshold = None
    using_google_drive = False

    render = False  # If you want to display the image, set this to True. Turn this off if you run this in Google Collab
   
    n_rollout_step_update = 5000

    n_update = 8000 # How many ppo update you want to run
    n_init_episode = 1024 # default 1014
    num_envs = 4
    seed = 450
    #############################################


    gym_id = "nasim:Medium-v0"
    log_dir = "./vevc_ppo_rnd/runs/" + gym_id + "/" "NUM_ENV_" + str(num_envs) + "/"
    #writer =  SummaryWriter(logdir=log_dir)
    writer =  None

    envs = VecPyTorch(DummyVecEnv([make_env(gym_id, seed+i*100, i, wrapper=None) for i in range(num_envs)]), device)

    test_envs = make_test_env(gym_id, 145, wrapper=None)
    action_dim = envs.action_space.n
    state_dim = envs.observation_space.shape  # .n
    print("Action Dimension is {}".format(action_dim))
    print("State Dimension is {}".format(state_dim))
    
    agent = PPO_Agent(state_dim, action_dim, hyper_params=PPO_HYPERPARAMS, is_training_mode=True, device=device)
    rollout_runner = RolloutRunner(state_dim, device=device)
    #############################################
    if using_google_drive:
        from google.colab import drive
        drive.mount('/test')

    if load_weights:
        agent.load_weights()
        print('Weight Loaded')

    rewards = []
    batch_rewards = []
    times = []
    batch_times = []

    t_updates = 0

    #############################################
    state = envs.reset()

    if training_mode:
        rollout_runner = run_inits_steps(envs, rollout_runner, render, n_init_episode)

    global_rollout_step = 0
    #############################################

    for i_update in range(1, n_update + 1):    
        
        rollout_runner, rollout_end_state, rollout_end_dones, global_rollout_step = rollout_steps(envs, agent, rollout_runner, n_rollout_step_update,\
             render, training_mode, state,  global_rollout_step)
        
        state = rollout_end_state
        next_done_flag = rollout_end_dones
        
        """
        we need to use two modules to get the job done
        1) Runner function:
            - do the rollouts
            - caculate advantage --> loss
        2) Learner
            - do the backprop
            - update ppo
            - update rnd
        """
        
        if not writer:
            rollout_runner = agent.update_ppo(rollout_runner, next_done_flag, writer=writer)
        else:
            rollout_runner,  pg_loss, critic_loss, dist_entropy = agent.update_ppo(rollout_runner, next_done_flag, writer=writer)
            writer.add_scalar('dist_entropy', dist_entropy, global_rollout_step)
            writer.add_scalar('critic_loss', critic_loss, global_rollout_step)
            writer.add_scalar('pg_loss', pg_loss, global_rollout_step)
           
        if i_update % 15 == 0 and i_update != 0:
            rollout_runner = agent.update_rnd(rollout_runner)
        
        '''
        Test trained agent with seperate env
        '''
        if i_update % 5 == 0 and i_update != 0:
            agent.is_training=False
            all_episode_reward = []
            all_episode_steps = []
            for _ in range(3):
                test_envs.reset()
                reward_per_eps, step_per_eps = test_episode(test_envs, agent)
                all_episode_reward.append(reward_per_eps)
                all_episode_steps.append(step_per_eps)
            print("TESTING: Episode {} \t t_reward: {: .2f} \t num_step: {: .2f} \t ".format(global_rollout_step, np.mean(all_episode_reward), np.mean(all_episode_steps)))
            if writer is not None:
                writer.add_scalar('rewards', np.mean(all_episode_reward), global_rollout_step)
                writer.add_scalar('number of steps', np.mean(all_episode_steps), global_rollout_step)
            
            agent.is_training=True

        
    print('========== Final ==========')
    writer.close()

    # Plot the reward, times for every episode
    for reward in batch_rewards:
        rewards.append(reward)

    for time in batch_times:
        times.append(time)

if __name__ == '__main__':
    main()

    
