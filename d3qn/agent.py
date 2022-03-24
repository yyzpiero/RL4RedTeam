import sys
from pathlib import Path

cwd = str(Path(__file__).parent)
sys.path.insert(0, cwd)
from control import Controller
from collections import namedtuple
import numpy as np
import torch as torch
import gym
import time
import math
from buffer import ReplayBuffer


class AgentBuild:

    rewards = []
    total_reward = 0
    birth_time = 0
    n_iter = 0
    n_games = 0

    Memory = namedtuple(
        "Memory",
        ["obs", "action", "new_obs", "reward", "done"],
        # verbose=False, - Python 3.7 removed this args
        rename=False,
    )

    # learning_rate, type='Adam', gamma=0.9
    # self.Controller.optimizer_build(hyperparameters['learning_rate'])

    def __init__(
        self,
        env: gym.envs,
        device: torch.device,
        hyperparameters: dict,
        summary_writer: bool = None,
    ) -> None:
        # print(env)
        self.controller = Controller(
            env.observation_space.shape,
            env.action_space.n,
            gamma=hyperparameters["gamma"],
            net_arch=hyperparameters["net_arch"],
            c51=hyperparameters["c51"],
            dueling=hyperparameters["dueling"],
            noisy_net=hyperparameters["noisy_net"],
            batch_size=hyperparameters["batch_size"],
            double_DQN=hyperparameters["double_DQN"],
            Vmax=hyperparameters["Vmax"],
            Vmin=hyperparameters["Vmin"],
            num_atoms=hyperparameters["num_atoms"],
            n_multi_step=hyperparameters["n_multi_step"],
            device=device,
        )
        # self.controller.set_optimizer(hyperparameters['learning_rate'])
        self.controller.optimizer_build(
            hyperparameters["learning_rate"],
            optim_type=hyperparameters["optimizer_type"],
            gamma=0.9,
        )

        self.birth_time = time.time()

        self.iter_update_target = hyperparameters["n_iter_update_target"]
        self.buffer_start_size = hyperparameters["buffer_start_size"]
        self.c51 = hyperparameters["c51"]
        self.dueling = hyperparameters["dueling"]
        self.epsilon_start = hyperparameters["epsilon_start"]
        self.epsilon = hyperparameters["epsilon_start"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_final = hyperparameters["epsilon_final"]

        self.accumulated_loss = []
        self.device = device

        # initialize the replay buffer (i.e. the memory) of the agent
        self.replay_buffer = ReplayBuffer(
            hyperparameters["buffer_capacity"],
            hyperparameters["n_multi_step"],
            hyperparameters["gamma"],
        )
        self.summary_writer = summary_writer

        self.noisy_net = hyperparameters["noisy_net"]

        self.env = env

    def act(self, obs):
        return self.controller.get_max_action(obs, self.dueling)

    def e_greedy_act(self, obs):
        """
        E-greedy action
        """

        # In case of a noisy net, it takes a greedy action
        if self.noisy_net:
            return self.act(obs)

        else:
            if np.random.random() > self.epsilon:  # or self.noisy_net:
                return self.act(obs)
            else:
                return self.env.action_space.sample()

    def add_env_feedback(self, obs, action, new_obs, reward, done):
        """
        Acquire a new feedback from the environment. The feedback is constituted by the new observation, the reward and the done boolean.
        """
        # Create the new memory and update the buffer
        new_memory = self.Memory(
            obs=obs, action=action, new_obs=new_obs, reward=reward, done=done
        )
        self.replay_buffer.append(new_memory)

        # update the variables
        self.n_iter += 1
        # decrease epsilon

        # self.epsilon = max(
        #     self.epsilon_final,
        #     self.epsilon_start - self.n_iter / self.epsilon_decay)

        # exponential decrease epsilon
        self.epsilon = self.epsilon_final + (
            self.epsilon_start - self.epsilon_final
        ) * (math.exp(-1.0 * self.n_iter / self.epsilon_decay))

        self.total_reward += reward

    def train(self, batch_size):
        """
        Sample batch_size memories from the buffer and optimize them
        """
        # self._update_learning_rate(self.optimizer)
        # optimizer = self.optimizer
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1, verbose=False)

        if len(self.replay_buffer) > self.buffer_start_size:
            # sample
            mini_batch = self.replay_buffer.sample(batch_size)
            # optimize
            l_loss = self.controller.optimize(mini_batch, batch_size)
            self.accumulated_loss.append(l_loss)

        # update target NN
        if self.n_iter % self.iter_update_target == 0:
            self.controller.update_target()

    def reset_stats(self):
        """
        Reset the agent's statistics
        """
        self.rewards.append(self.total_reward)
        self.total_reward = 0
        self.accumulated_loss = []
        self.n_games += 1

    # def reset_iter(self):
    #     """
    #     Reset the agent's statistics
    #     """
    #     self.n_games += 1

    def print_info(self, verbose=0):
        """
        Print information about the agent
        """
        # fps = (self.n_iter-self.ts_frame)/(time.time()-self.ts)
        if verbose == 1:
            print(
                "iteration:%d episode:%d rew_per_episode:%.2f mean_rew:%.2f eps:%.2f, loss:%.4f"
                % (
                    self.n_iter,
                    self.n_games,
                    self.total_reward,
                    np.mean(self.rewards[-40:]),
                    self.epsilon,
                    np.mean(self.accumulated_loss),
                )
            )

        # self.ts_frame = self.n_iter
        # self.ts = time.time()

        if self.summary_writer != None:
            # self.summary_writer.add_scalar('reward', self.total_reward,
            #                               self.n_games)
            # self.summary_writer.add_scalar('mean_reward',
            #                               np.mean(self.rewards[-40:]),
            #                               self.n_games)
            # self.summary_writer.add_scalar('10_mean_reward',
            #                               np.mean(self.rewards[-10:]),
            #                               self.n_games)
            self.summary_writer.add_scalar("esilon", self.epsilon, self.n_games)
            self.summary_writer.add_scalar(
                "loss", np.mean(self.accumulated_loss), self.n_games
            )
