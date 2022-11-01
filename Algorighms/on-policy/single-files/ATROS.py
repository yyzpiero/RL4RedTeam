import argparse
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from tensorboardX import SummaryWriter
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from utils import make_env, layer_init, matrix_norm

def parse_args():
    #fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=13,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with tensorboard")
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="nasim:CyberRange-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=12,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=256,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
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
    parser.add_argument("--hidden-size", type=int, default=512,
        help="hidden layer size of the neural networks")
    parser.add_argument("--coverage", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use coverage mechanism as per the paper.")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

class Agent_Memory(nn.Module):
    '''
    When RNN policy is enabled, critic and actor share the first several input/preception layers.
    '''
    def __init__(self, envs, hidden_size):
        
        super().__init__()
        self.shared_network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size)
        
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.zeros_(param.data)
            elif "weight" in name:
                nn.init.xavier_uniform_(param.data)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, envs.action_space.n), std=0.01)
        )
        self.coverage = nn.Sequential(
            layer_init(nn.Linear(envs.action_space.n, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, envs.action_space.n), std=0.01)
        )    
        self.cw_learner = nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, envs.action_space.n), std=0.01)
        )
        self.shared_learner=nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(),hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size,np.array(envs.observation_space.shape).prod()),std=0.01)
        )
        self.actionN=nn.Sequential(
            layer_init(nn.Linear(1,hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size,np.array(envs.observation_space.shape).prod()),std=0.01)
        )

    def get_states(self, lstm_state, done,next_belief_actor_tmp,next_obs):
       
        f_obs=self.shared_learner(next_obs)
        f_belief_state=self.shared_learner(next_belief_actor_tmp)
        w_obs=self.shared_learner(next_obs)
        w_belief_state=self.shared_learner(next_belief_actor_tmp)
        next_bs_=(torch.mul(f_obs,w_obs)+torch.mul(f_belief_state,w_belief_state))
        hidden = self.shared_network(next_bs_)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
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
        return new_hidden, lstm_state,next_bs_
    
    def get_value(self,next_obs,lstm_state, done,next_belief_actor_tmp):
        hidden, _,typed = self.get_states(lstm_state, done,next_belief_actor_tmp,next_obs)
        return self.critic(hidden)

    def get_action_and_value(self,next_obs,lstm_state, done,next_belief_actor_tmp,action=None, coverage_hist=None):

        hidden, lstm_state,next_bs_ = self.get_states(lstm_state, done,next_belief_actor_tmp,next_obs) 
        if coverage_hist is None:
            logits = self.actor(hidden)
        else:
            logits = self.actor(hidden)
            c_w = self.cw_learner(hidden)
            c_out = self.coverage(coverage_hist)
            logits = logits + c_w * c_out

        action_probs = Categorical(logits=logits)

        if action is None:
            action_N = action_probs.sample()
            action_tem=action_N.reshape(-1,1).type(torch.float32)
            action_Mlp=self.actionN(action_tem)
            next_bs_tmp=action_Mlp+next_bs_
            return action_N, action_probs.log_prob(action_N), action_probs.entropy(), self.critic(hidden), lstm_state,next_bs_tmp,next_bs_
        else:
            action_Y=action_probs.probs
            action_N = action_probs.sample()
            action_tem=action.reshape(-1,1).type(torch.float32)
            action_Mlp=self.actionN(action_tem)
            next_bs_tmp=action_Mlp+next_bs_
            return action_Y, action_probs.log_prob(action), action_probs.entropy(), self.critic(hidden), lstm_state,next_bs_tmp,next_bs_


def update_dict(file_path1, file_path2, ordered_length, ordered_return, algorithm ):
    if(os.path.getsize(file_path1)!=0):
        length_dict=np.load(file_path1,allow_pickle=True).item()
        if(algorithm in length_dict.keys()):
            length_dict[algorithm]=torch.cat((length_dict[algorithm], ordered_length), 1)
            np.save(file_path1,length_dict)
        else:
            dict_tmp={algorithm:ordered_length}
            dict_length=dict(length_dict,**dict_tmp)
            np.save(file_path1,dict_length)
    else:
        nasim_len_dict={algorithm:ordered_length}
        np.save(file_path1,nasim_len_dict)
    if(os.path.getsize(file_path2)!=0):
        reward_dict=np.load(file_path2,allow_pickle=True).item()
        if(algorithm in reward_dict.keys()):
            reward_dict[algorithm]=torch.cat((reward_dict[algorithm], ordered_return), 1)
            np.save(file_path2,reward_dict)
        else:
            dict_tmp={algorithm:ordered_return}
            dict_reward=dict(reward_dict,**dict_tmp)
            np.save(file_path2,dict_reward)
    else:
        nasim_reward_dict={algorithm:ordered_return}
        np.save(file_path2,nasim_reward_dict)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{args.num_steps}__{args.hidden_size}__{int(time.time())}"
    
    if args.track:
        writer = SummaryWriter(f"./runs/{run_name}")
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

#    temp_time=time.time()
    envs = DummyVecEnv([make_env(args.env_id, args.seed + i, i) for i in range(args.num_envs)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True)
#    end_time = time.time()
#    print("Dummy_Vec_Env: ",end_time-temp_time)

    agent = Agent_Memory(envs, args.hidden_size).to(device)
    total_para_nums = sum([param.nelement() for param in agent.parameters()])
    print("Number of parameter: %.2fM",total_para_nums/1e6)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    belief_actor = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    belief_actor_tmp = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ebs=torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    episode_step = torch.zeros(args.num_envs)
    training_episode_step = torch.zeros(args.num_envs)
    reward_training_max = torch.zeros(args.num_envs)
    steps_training_min = torch.zeros(args.num_envs)+np.inf
    reward_training_episode = torch.zeros(10,args.num_envs)-np.inf
    reward_training_episode_length = torch.zeros(10,args.num_envs)+np.inf
    start_time = time.time()
    envs.reset()
#    envs.render(mode="readable")
    algorithm='ATROS(ours)'
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_belief_actor = next_obs.clone().to(device)
    next_belief_actor_tmp = next_obs.clone().to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )  # hidden and cell states 
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        if args.coverage:
            coverage_hist = np.zeros((args.num_envs, envs.action_space.n))
        else:
            coverage_hist = None

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            belief_actor[step] = next_belief_actor
            belief_actor_tmp[step] = next_belief_actor_tmp
            dones[step] = next_done
            episode_step += 1
            # ALGO LOGIC: action logic
            with torch.no_grad():
                if coverage_hist is None:
                    action, logprob, _, value, next_lstm_state,next_belief_actor_tmp,next_belief_actor = agent.get_action_and_value(next_obs,next_lstm_state, next_done,next_belief_actor_tmp)
                else:
                    coverage_hist_norm = matrix_norm(coverage_hist)
                    coverage_hist_norm = torch.FloatTensor(coverage_hist_norm).to(device).detach()
                    action, logprob, _, value, next_lstm_state,next_belief_actor_tmp,next_belief_actor = agent.get_action_and_value(next_obs,next_lstm_state, next_done,next_belief_actor_tmp,coverage_hist=coverage_hist_norm)
                    coverage_hist[range(args.num_envs), action.cpu()] += 1 
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            # TRY NOT TO MODIFY: execute the game and log data.
            actions_for_envs_temp = action.cpu().numpy()
            actions_for_envs = [int(i) for i in actions_for_envs_temp]
            next_obs, reward, done, info = envs.step(actions_for_envs)

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for num_e in range(0, args.num_envs):
                item = info[num_e]
                if "episode" in item.keys():
                    training_episode_step[num_e] += 1
                    episode_step[num_e]=0
                    reward_coveraged_episode = float(item['episode']['r'])
                    steps_coveraged_episode = int(item['episode']['l'])
                    reward_training_episode[int(training_episode_step[num_e])%10-1,num_e] = reward_coveraged_episode
                    reward_training_episode_length[int(training_episode_step[num_e])%10-1,num_e]=steps_coveraged_episode
                    ave_episode_return = torch.mean(reward_training_episode[:,num_e])
                    ave_episode_length = torch.mean(reward_training_episode_length[:,num_e])
                    ordered_episode_return = torch.sort(torch.mean(reward_training_episode,0))[0]
                    ordered_episode_length = torch.sort(torch.mean(reward_training_episode_length,0))[0]

                    quartile_mean_episode_return = torch.mean(ordered_episode_return[int((args.num_envs+1)/4):int((args.num_envs+1)*3/4)])
                    quartile_mean_episode_length = torch.mean(ordered_episode_length[int((args.num_envs+1)/4):int((args.num_envs+1)*3/4)])
                    ordered_return=ordered_episode_return.reshape([args.num_envs,1])
                    ordered_length=ordered_episode_length.reshape([args.num_envs,1])

                    if global_step>1500000-int(args.num_steps/2)-1 and global_step<1500000+int(args.num_steps/2) and (args.env_id=='nasim:LargeGen-PO-v0'):
                        file_path1='./output-file/LargeGen_1M_length.npy'
                        file_path2='./output-file/LargeGen_1M_reward.npy'
                        update_dict(file_path1,file_path2,ordered_length,ordered_return,algorithm)
                    if global_step>3000000-int(args.num_steps/2)-1 and global_step<3000000+int(args.num_steps/2) and (args.env_id=='nasim:LargeGen-PO-v0'):
                        file_path1='./output-file/LargeGen_3M_length.npy'
                        file_path2='./output-file/LargeGen_3M_reward.npy'
                        update_dict(file_path1,file_path2,ordered_length,ordered_return,algorithm)

                    if reward_coveraged_episode > reward_training_max[num_e]-50 or (reward_coveraged_episode==reward_training_max[num_e] and steps_coveraged_episode < steps_training_min[num_e]):
                        optimal_actions = actions[(step+1-steps_coveraged_episode):(step+1),num_e].reshape(-1)
                        print(f"current_env={num_e},global_step={global_step}, episodic_return={reward_coveraged_episode}, episodic_length={steps_coveraged_episode}")
                        print("Actions: ",optimal_actions)
                        reward_training_max[num_e] = max(reward_coveraged_episode,reward_training_max[num_e])
                        steps_training_min[num_e] = steps_coveraged_episode
                    if (writer is not None) and (int(min(training_episode_step))>9):
                        name_return="charts/episodic_return_of_num_envs_"+str(num_e)
                        name_length="charts/episodic_length_of_num_envs_"+str(num_e)
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        writer.add_scalar(name_return, ave_episode_return, training_episode_step[num_e])
                        writer.add_scalar(name_length, ave_episode_length, training_episode_step[num_e])
                        writer.add_scalar("charts/IQM_episodic_return", quartile_mean_episode_return, global_step)
                        writer.add_scalar("charts/IQM_episodic_length", quartile_mean_episode_length, global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
                next_belief_actor_tmp
            ).reshape(1, -1)
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
        b_belief_actor = belief_actor.reshape((-1,) + envs.observation_space.shape)
        b_belief_actor_tmp=belief_actor_tmp.reshape((-1,) + envs.observation_space.shape)
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
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                new_a_prob, newlogprob, entropy, newvalue, _,new_belief_actor_tmp,new_belief_actor = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_belief_actor_tmp[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl
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
                Mse_loss=torch.nn.MSELoss()               
                bstmp_obs_loss=Mse_loss(b_obs[mb_inds], new_belief_actor_tmp)
                entropy_loss = entropy.mean()
                if coverage_hist is None:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef+bstmp_obs_loss*0.5
                else:    
                    coverage_loss = torch.mean(torch.min(torch.mean(coverage_hist_norm, 0) , torch.mean(new_a_prob, 0)))
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + coverage_loss * 0.025 + bstmp_obs_loss *0.5
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if writer is not None:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("losses/bstmp_obs_loss", bstmp_obs_loss.item(), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        #print("SPS:", int(global_step / (time.time() - start_time)))
        #print("Total time:",time.time()-start_time)
    envs.close()
    if writer is not None:
        writer.close()
