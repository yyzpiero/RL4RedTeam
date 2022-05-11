from Agents.dqn_agent import DQN_Agent, DQN_C51Agent
from Agents import MultiPro
from Wrapper import wrapper_new
import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse 
import time

def evaluate( eps, frame, eval_runs=5):
    """
    Makes an evaluation runs with eps 0.001 
    """
    reward_batch = []
    for i in range(eval_runs):
        state = eval_env.reset()
        rewards = 0
        while True:
            action = agent.act(np.expand_dims(state, axis=0), 0.001, eval=True)
            state, reward, done, _ = eval_env.step(action[0].item())
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
        
    writer.add_scalar("Reward", np.mean(reward_batch), frame)


def run_random_policy(random_frames):
    """
    Run env with random policy for x frames to fill the replay memory.
    """
    state = eval_env.reset() 
    for i in range(random_frames):
        action = np.random.randint(action_size)
        next_state, reward, done, _ = eval_env.step(action)
        agent.memory.add(state, action, reward, next_state, done)
        next_state = state
        if done:
            state = eval_env.reset()

def run(frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.01, eval_every=1000, eval_runs=5, worker=1):
    """Deep Q-Learning.
    
    Params
    ======
        frames (int): maximum number of training frames
        eps_fixed (bool): training with greedy policy and noisy layer (fixed) or e-greedy policy (not fixed)
        eps_frames (float): number of frames to decay epsilon exponentially
        min_eps (float): minimum value of epsilon from where eps decays linear until the last frame
        eval_every (int): number frames when evaluation runs are done
        eval_runs (int): number of evaluation runs
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    d_eps = eps_start - min_eps
    i_episode = 1
    state = envs.reset()
    score = 0                  
    for frame in range(1, frames+1):

        action = agent.act(state, eps)
        next_state, reward, done, _ = envs.step(action)
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            agent.step(s, a, r, ns, d, writer)
        state = next_state
        score += reward
        # linear annealing to the min epsilon value until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            eps = max(eps_start - ((frame*d_eps)/eps_frames), min_eps)


        # evaluation runs
        if frame % eval_every == 0 or frame == 1:
            evaluate(eps, frame*worker, eval_runs)

        if done.any():
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            writer.add_scalar("Average100", np.mean(scores_window), i_episode)
            print('\rEpisode {}\tFrame {} \tAverage Score: {:.2f}'.format(i_episode*worker, frame*worker, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage Score: {:.2f}'.format(i_episode*worker, frame*worker, np.mean(scores_window)))
            i_episode +=1 
            state = envs.reset()
            score = 0              

    return np.mean(scores_window)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-agent", type=str, choices=["dqn",
                                                     "dqn+per",
                                                     "noisy_dqn",
                                                     "noisy_dqn+per",
                                                     "dueling",
                                                     "dueling+per", 
                                                     "noisy_dueling",
                                                     "noisy_dueling+per", 
                                                     "c51",
                                                     "c51+per", 
                                                     "noisy_c51",
                                                     "noisy_c51+per", 
                                                     "duelingc51",
                                                     "duelingc51+per", 
                                                     "noisy_duelingc51",
                                                     "noisy_duelingc51+per",
                                                     "rainbow" ], default="dqn", help="Specify which type of DQN agent you want to train, default is DQN - baseline!")
    
    parser.add_argument("-env", type=str, default="PongNoFrameskip-v4", help="Name of the atari Environment, default = Pong-v0")
    parser.add_argument("-frames", type=int, default=int(5e6), help="Number of frames to train, default = 5 mio")
    parser.add_argument("-seed", type=int, default=1, help="Random seed to replicate training runs, default = 1")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, help="Batch size for updating the DQN, default = 32")
    parser.add_argument("-layer_size", type=int, default=512, help="Size of the hidden layer, default=512")
    parser.add_argument("-n_step", type=int, default=1, help="Multistep DQN, default = 1")
    parser.add_argument("-m", "--memory_size", type=int, default=int(1e5), help="Replay memory size, default = 1e5")
    parser.add_argument("-lr", type=float, default=0.00025, help="Learning rate, default = 0.00025")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount factor gamma, default = 0.99")
    parser.add_argument("-t", "--tau", type=float, default=1e-3, help="Soft update parameter tat, default = 1e-3")
    parser.add_argument("-eps_frames", type=int, default=1000000, help="Linear annealed frames for Epsilon, default = 1mio")
    parser.add_argument("-eval_every", type=int, default=50000, help="Evaluate every x frames, default = 50000")
    parser.add_argument("-eval_runs", type=int, default=5, help="Number of evaluation runs, default = 5")
    parser.add_argument("-min_eps", type=float, default = 0.1, help="Final epsilon greedy value, default = 0.1")
    parser.add_argument("-ic", "--intrinsic_curiosity", type=int, choices=[0,1,2], default=0, help="Adding intrinsic curiosity to the extrinsic reward. 0 - only reward and no curiosity, 1 - reward and curiosity, 2 - only curiosity, default = 0")
    parser.add_argument("-info", type=str, help="Name of the training run")
    parser.add_argument("--fill_buffer", type=int, default=50000, help="Adding samples to the replay buffer based on a random policy, before agent-env-interaction. Input numer of preadded frames to the buffer, default = 50000")
    parser.add_argument("-w", "--worker", type=int, default=1, help="Number of parallel working environments, default is 1")
    parser.add_argument("-save_model", type=int, choices=[0,1], default=1, help="Specify if the trained network shall be saved or not, default is 1 - saved!")

    args = parser.parse_args()
    if args.agent == "rainbow":
        args.n_step = 2
        args.agent = "noisy_duelingc51+per"

    writer = SummaryWriter("runs/"+str(args.info))

    BUFFER_SIZE = args.memory_size
    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    TAU = args.tau
    LR = args.lr
    seed = args.seed
    n_step = args.n_step
    env_name = args.env
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    torch.autograd.set_detect_anomaly(True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if "-ram" in args.env or args.env == "CartPole-v0" or args.env == "LunarLander-v2": 
        envs = MultiPro.SubprocVecEnv([lambda: gym.make(args.env) for i in range(args.worker)])
        eval_env = gym.make(args.env)
    else:
        envs = MultiPro.SubprocVecEnv([lambda: wrapper_new.make_env(args.env) for i in range(args.worker)])
        eval_env = wrapper_new.make_env(args.env)
    envs.seed(seed)
    eval_env.seed(seed+1)

    action_size     = eval_env.action_space.n
    state_size = eval_env.observation_space.shape

    if not "c51" in args.agent:
        agent = DQN_Agent(state_size=state_size,    
                        action_size=action_size,
                        Network=args.agent,
                        layer_size=args.layer_size,
                        n_step=n_step,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE, 
                        LR=LR, 
                        TAU=TAU, 
                        GAMMA=GAMMA, 
                        curiosity=args.intrinsic_curiosity, 
                        worker=args.worker,
                        device=device, 
                        seed=seed)
    else:
        agent = DQN_C51Agent(state_size=state_size,
                        action_size=action_size,
                        Network=args.agent, 
                        layer_size=args.layer_size,
                        n_step=n_step,
                        BATCH_SIZE=BATCH_SIZE, 
                        BUFFER_SIZE=BUFFER_SIZE, 
                        LR=LR, 
                        TAU=TAU, 
                        GAMMA=GAMMA, 
                        curiosity=args.intrinsic_curiosity,
                        worker=args.worker,
                        device=device, 
                        seed=seed)
                        
    # adding x frames of random policy to the replay buffer before training!
    if args.fill_buffer != None:
        run_random_policy(args.fill_buffer)
        print("Buffer size: ", agent.memory.__len__())

    # set epsilon frames to 0 so no epsilon exploration
    if "noisy" in args.agent:
        eps_fixed = True
    else:
        eps_fixed = False

    t0 = time.time()
    final_average100 = run(frames = args.frames//args.worker, eps_fixed=eps_fixed, eps_frames=args.eps_frames//args.worker, min_eps=args.min_eps, eval_every=args.eval_every//args.worker, eval_runs=args.eval_runs, worker=args.worker)
    t1 = time.time()
    
    print("Training time: {}min".format(round((t1-t0)/60,2)))
    if args.save_model:
        torch.save(agent.qnetwork_local.state_dict(), args.info+".pth")

    hparams = {"agent": args.agent,
               "batch size": args.batch_size*args.worker,
               "layer size": args.layer_size, 
               "n_step": args.n_step,
               "memory size": args.memory_size,
               "learning rate": args.lr,
               "gamma": args.gamma,
               "soft update tau": args.tau,
               "epsilon decay frames": args.eps_frames,
               "min epsilon": args.min_eps,
               "random warmup": args.fill_buffer}
    metric = {"final average 100 reward": final_average100}
    writer.add_hparams(hparams, metric)
