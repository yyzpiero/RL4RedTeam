# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5"
import random
import copy
import time
from distutils.util import strtobool
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tensorboardX import SummaryWriter
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from utils import make_env, layer_init, make_env_list_random, FastGLU, Transformer

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=56747,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with tensorboard")
    # parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
    #     help="the wandb's project name")
    # parser.add_argument("--wandb-entity", type=str, default=None,
    #     help="the entity (team) of wandb's project")
    # parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    #     help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="nasim:MediumPO2D-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1024,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=20,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--hidden-size", type=int, default=256,
        help="hidden layer size of the neural networks")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

class Agent(nn.Module):
    def __init__(self, envs, hidden_size):
        super().__init__()
        self.transformer = Transformer(d_model=hidden_size, d_inner=256,
                                       n_layers=3, n_head=2, d_k=128, 
                                       d_v=128, 
                                       dropout=0.)  # make dropout=0 to make training and testing consistent

        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0)
        )
        
        self.share_preception = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape)[-1], hidden_size)),
            nn.Tanh()
        )
        
        self.actor = nn.Sequential(    
            # layer_init(nn.Linear(hidden_size, hidden_size)),
            # nn.ReLU(),
            FastGLU(hidden_size),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, envs.action_space.n), std=0.01)
        )

        self.preception_actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape)[-1], hidden_size)),
            nn.Tanh()
        )
        
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=1, stride=1,
                               padding=0, bias=False)
        
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param.data)
            elif "weight" in name:
                nn.init.xavier_uniform_(param.data)
    
    def get_states(self, x, lstm_state, done):
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = x.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_shared(x, lstm_state, done)
        return self.critic(hidden)

    def get_shared(self, x, lstm_state, done):
        precepted = self.share_preception(x)
        trans_out = self.transformer(precepted)
        conv_out = F.relu(self.conv1(trans_out.transpose(-2, -1))).transpose(-1, -2) # IDK why, but AlphaStar uses this "trick", conv over 256 embedded features
        conv_out = conv_out.mean(dim=1, keepdim=False)
        hidden, lstm_state = self.get_states(conv_out, lstm_state, done)
        return hidden, lstm_state

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_shared( x, lstm_state, done)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state




def main():
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        writer = SummaryWriter(f"DefRuns/{run_name}")
        writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    device_ids = [0, 1]
    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #      [make_env(args.env_id, args.seed + i, i) for i in range(args.num_envs)]
    # )
    #env = DummyVecEnv([lambda:env])
    if args.env_id[0:7] == "nasim:c":
        envs_list = make_env_list_random(args.env_id, args.seed, args.num_envs)
        envs = DummyVecEnv(envs_list)
    else:
        envs = DummyVecEnv([make_env(args.env_id, args.seed + i, i) for i in range(args.num_envs)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True)
    #assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    agent = Agent(envs, args.hidden_size)
    agent = nn.DataParallel(agent, device_ids=device_ids)
    #.to(device)
    agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    if isinstance(agent, torch.nn.DataParallel):
        next_lstm_state = (
            torch.zeros(agent.module.lstm.num_layers, args.num_envs, agent.module.lstm.hidden_size).to(device),
            torch.zeros(agent.module.lstm.num_layers, args.num_envs, agent.module.lstm.hidden_size).to(device),
        ) 
    else:
        next_lstm_state = (
            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        ) 
    for update in range(1, num_updates + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.module.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if isinstance(agent, torch.nn.DataParallel):
                    action, logprob, _, value, next_lstm_state = agent.module.get_action_and_value(next_obs, next_lstm_state, next_done)
                else:
                    action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}, episodic_length={item['episode']['l']}")
                    if writer is not None:
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            if isinstance(agent, torch.nn.DataParallel):
                next_value = agent.module.get_value(next_obs,next_lstm_state,next_done).reshape(1, -1)
            else:
                next_value = agent.get_value(next_obs,next_lstm_state,next_done).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_dones = dones.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start +  envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index
                if isinstance(agent, torch.nn.DataParallel):
                    _, newlogprob, entropy, newvalue, _ = agent.module.get_action_and_value(
                            b_obs[mb_inds], 
                            (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                            b_dones[mb_inds],
                            b_actions.long()[mb_inds])
                else:
                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                            b_obs[mb_inds], 
                            (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                            b_dones[mb_inds],
                            b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                if isinstance(agent, torch.nn.DataParallel):
                    optimizer.module.step()
                else:    
                    optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if writer is not None:
            writer.add_scalar("charts/learning_rate", optimizer.module.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
            

    envs.close()
    if writer is not None:
        writer.close()

if __name__ == '__main__':
    since = time.time()
    main()
    time_elapsed = time.time()-since
    print("Total Run Time: {}".format(time_elapsed))