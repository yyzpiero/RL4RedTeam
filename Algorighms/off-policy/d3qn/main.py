import datetime
import torch
import json
import numpy as np
import gym
from gym_minigrid.wrappers import *

from environment import EnvironmentBuild
from agent import AgentBuild
from my_utils import get_device, test_game

from tensorboardX import SummaryWriter


# Main Hyper Parameteres
DQN_HYPERPARAMS = {
    'dueling': True,
    "c51": True,
    "num_atoms": 51,
    "double_DQN": True,
    "buffer_start_size": 5000,
    "buffer_capacity": 100000,
    "n_multi_step": 5,
    "noisy_net": False,
    "epsilon_start": 0.9,
    "epsilon_decay": 400000,
    "epsilon_final": 0.05,
    'Vmax': +100.0,
    "Vmin": -2000.0,
    "optimizer_type": "Adam",
    "learning_rate": 2.5e-4,
    "gamma": 0.99,
    "n_iter_update_target": 5000,
    "batch_size": 128,
    "net_arch": [256, 256],
    "env_name": "nasim:Tiny-v0"
}

DEVICE = torch.device(get_device("auto"))
ENV_NAME = DQN_HYPERPARAMS["env_name"]
MAX_N_GAMES = int(20000)
MAX_N_Iter = 1000000
LOG_DIR = "./content/runs/DQN/MultiSeed"

if DQN_HYPERPARAMS["env_name"].startswith('nasim'):
    file_name = '_'.join([DQN_HYPERPARAMS["env_name"][6:]])
else:
    file_name = '_'.join([DQN_HYPERPARAMS["env_name"][:]])

SUMMARY_WRITER = False
BATCH_SIZE = int(DQN_HYPERPARAMS["batch_size"])
TEST_PER_N_ITER = int(2000)
TEST_N_EPISODE = 1
SEED_NUM = 1
TRAIN_FREQ = 4


if __name__ == "__main__":

    env = EnvironmentBuild(ENV_NAME)

    env = gym.make('MiniGrid-DoorKey-5x5-v0')
    #env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = FlatObsWrapper(env) # Get rid of the 'mission' field
    test_env = EnvironmentBuild(ENV_NAME)
    test_env = gym.make('MiniGrid-DoorKey-5x5-v0')
    #env = RGBImgPartialObsWrapper(env) # Get pixel observations
    test_env = FlatObsWrapper(test_env) # Get rid of the 'mission' field

    obs = env.reset()
    test_env.reset()

    tb_log_dir = LOG_DIR + "/" + file_name + "/" + datetime.datetime.now().strftime("%m-%d-%H-%M")

    print("Hyperparams:", DQN_HYPERPARAMS)

    for seed in range(SEED_NUM):
        torch.manual_seed(seed * 10)
        np.random.seed(seed * 10)

        if SUMMARY_WRITER:
            writer = SummaryWriter('%s/seed-%s' % (tb_log_dir, seed))
            with open(tb_log_dir + "/" + "HyperParas.json", mode="w") as fp:
                json.dump(DQN_HYPERPARAMS, fp)
        else:
            writer = None

        agent = AgentBuild(env, device=DEVICE, summary_writer=writer, hyperparameters=DQN_HYPERPARAMS)

        n_games = 0
        x_loop_must_break = False

        for n_iter in range(MAX_N_Iter):

            obs = env.reset()
            done = False

            while not done:
                if agent.n_iter >= MAX_N_Iter:
                    x_loop_must_break = True
                    break

                action = agent.e_greedy_act(obs)
                new_obs, reward, done, _ = env.step(action)
                agent.add_env_feedback(obs, action, new_obs, reward, done)
                obs = new_obs

                if agent.n_iter % TRAIN_FREQ == 0:
                    agent.train(BATCH_SIZE)

                if agent.n_iter % TEST_PER_N_ITER == 0:
                    test_mean_reward, test_mean_reward_std, test_mean_step, test_mean_step_std = test_game(
                        test_env, agent, TEST_N_EPISODE)

                    agent.print_info(verbose=0)
                    print("Test: reward_per_episode_mean:{:.2f} at episode:{} with {} steps at itertaion:{}"
                        .format(test_mean_reward, agent.n_games, test_mean_step, agent.n_iter))
                    
                    if writer is not None:
                        writer.add_scalar("test_total_reward",test_mean_reward, agent.n_games)
                        writer.add_scalar("test_total_reward_iter", test_mean_reward, agent.n_iter)
                        writer.add_scalar("test_total_reward_iter_std", test_mean_reward_std, agent.n_iter)
                        writer.add_scalar("test_total_step_iter", test_mean_step, agent.n_iter)
                        writer.add_scalar("test_total_step_iter_std", test_mean_step_std, agent.n_iter)
                        writer.add_scalars("rewards", {"test_total_reward": test_mean_reward}, agent.n_iter)

            if x_loop_must_break == True:
                break

            agent.reset_stats()

        if writer is not None:
            writer.export_scalars_to_json('%s/seed-%s/' % (tb_log_dir, seed) + "all_scalars.json")
            writer.close()

# tensorboard --logdir content/runs --host localhost
