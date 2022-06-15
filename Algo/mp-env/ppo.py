from model import Agent, RolloutStorage
import torch.optim as optim
import numpy as np
import torch


class PPO():
    def __init__(self, envs, num_envs, actor_critic, num_steps, hidden_size,learning_rate, batch_size, clip_param, ppo_epoch, num_mini_batch, value_loss_coef, entropy_coef,
                 lr=None, eps=None, max_grad_norm=None, use_clipped_value_loss=True, recompute_returns=False, tb_writer,
                 use_gae=True, gamma=0.99, gae_lambda=0.95, device='auto'):

        # Constants
        self.num_envs = num_envs 
        self.observation_space_shape = envs.observation_sapce.shape
        self.action_space_shape = envs.action_space.shape

        
        self.minibatch_size = 
        self.epochs
        self.num_steps = num_steps

        # Models
        self.agent = Agent(envs, hidden_size)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=learning_rate, eps=1e-5)
        self.rollouts = RolloutStorage(num_steps, num_envs=self.num_envs, observation_space_shape=self.observation_space_shape, 
                        action_space_shape=self.action_space_shape, device=self.device)
        
        # Writer
        self.writer = tb_writer
        self.global_step = 0

    def train(self, total_timesteps):
        self.batch_size = int(self.num_envs * self.num_steps)
        _num_updates = total_timesteps // self.batch_size
        next_obs = torch.Tensor(self.envs.reset()).to(device)
        next_done = torch.zeros(self.num_envs).to(device)

        for update in range(1, _num_updates + 1):
            
            # 1. Annealing the learning rate is instructed
            if self.anneal_lr:
                _frac = 1.0 - (update - 1.0) / _num_updates
                lrnow = _frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # 2. Collect Rollouts
            for step in range(self.num_steps):
                self.global_step += 1 * self.num_envs
                self.rollouts.obs[step] = next_obs
                self.rollouts.dones[step] = next_done

            # Update actor critic parameters from batches

            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for e in range(self.epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

    def eval(self, env):

    def get_parameters():

    def set_parameters(load_path_or_dict, exact_match=True, device='auto'):