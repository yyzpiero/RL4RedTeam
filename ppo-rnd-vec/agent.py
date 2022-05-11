import torch
import numpy as np

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, TensorDataset
from itertools import cycle

from networks import *
from distribution import *
from memory import *
from utili import *


class PolicyFunction():
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam

    def monte_carlo_discounted(self, rewards, dones):
        running_add = 0
        returns = []

        for step in reversed(range(len(rewards))):
            running_add = rewards[step] + \
                (1.0 - dones[step]) * self.gamma * running_add
            returns.insert(0, running_add)

        return torch.stack(returns)

    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value
        return q_values

    def generalized_advantage_estimation(self, values, rewards, next_values, dones):
        gae = 0
        adv = []

        delta = rewards + (1.0 - dones) * self.gamma * next_values - values
        for step in reversed(range(len(rewards))):
            gae = delta[step] + (1.0 - dones[step]) * \
                self.gamma * self.lam * gae
            adv.insert(0, gae)

        return torch.stack(adv)



class PPO_Agent():
    def __init__(self, state_dim, action_dim, hyper_params, is_training_mode, device):
    
    # state_dim, action_dim, is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
    #              minibatch, PPO_epochs, gamma, lam, learning_rate, mask_type, coverage, coverage_coef):
        self.policy_kl_range = hyper_params["policy_kl_range"]
        self.policy_params = hyper_params["policy_params"]
        self.learning_rate = hyper_params["learning_rate"]
        self.value_clip = hyper_params["value_clip"]
        self.entropy_coef = hyper_params["entropy_coef"]
        self.vf_loss_coef = hyper_params["vf_loss_coef"]
        self.gamma = hyper_params["gamma"]
        self.lam = hyper_params["lam"]
        self.coverage_coef = hyper_params["coverage_coef"]
        #self.rnd_step_update = hyper_params["rnd_step_update"]
        self.minibatch = hyper_params["minibatch"]
        self.PPO_epochs = hyper_params["PPO_epochs"]
        self.RND_epochs = 5
        self.is_training_mode = is_training_mode
        self.action_dim = action_dim
        self.action_mask = []
        self.mask_type = hyper_params["mask_type"]
        self.coverage = hyper_params["coverage"]
        self.device = device
        #self.num_envs = hyper_params["num_envs"]
        self.num_hidden = hyper_params["num_hidden"]
        self.anneal_lr = hyper_params["anneal_lr"]

        if self.coverage:
            self.actor = Actor_Model_Coverage(state_dim, action_dim, self.num_hidden, self.mask_type)
            self.actor_old = Actor_Model_Coverage(state_dim, action_dim, self.num_hidden, self.mask_type)
        else:
            self.actor = Actor_Model(state_dim, action_dim, self.mask_type, self.num_hidden)
            self.actor_old = Actor_Model(state_dim, action_dim, self.mask_type, self.num_hidden)        
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.learning_rate, eps=1e-5)

        self.ex_critic = Critic_Model(state_dim, action_dim, self.num_hidden)
        self.ex_critic_old = Critic_Model(state_dim, action_dim, self.num_hidden)
        self.ex_critic_optimizer = Adam(self.ex_critic.parameters(), lr=self.learning_rate, eps=1e-5)

        self.in_critic = Critic_Model(state_dim, action_dim, self.num_hidden)
        self.in_critic_old = Critic_Model(state_dim, action_dim, self.num_hidden)
        self.in_critic_optimizer = Adam(self.in_critic.parameters(), lr=self.learning_rate, eps=1e-5)

        self.rnd_predict = RND_Model(state_dim, action_dim, self.num_hidden)
        self.rnd_predict_optimizer = Adam(self.rnd_predict.parameters(), lr=self.learning_rate, eps=1e-5)
        self.rnd_target = RND_Model(state_dim, action_dim, self.num_hidden)
        for param in self.rnd_target.parameters():
            param.requires_grad = False

        #self.memory = Memory()
        #self.obs_memory = ObsMemory(state_dim)

        self.reward_rms = RunningMeanStd()
        #obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
        self.obs_rms = RunningMeanStd()
        self.discounted_reward = RewardForwardFilter(self.gamma)

        self.policy_function = PolicyFunction(self.gamma, self.lam)
        if self.mask_type == "Soft":
            self.distributions = DistributionsMasked()
        else:
            self.distributions = Distributions()
        #self.distributions = DistributionsMasked()
        self.utils = Utils()

        if self.anneal_lr:
        # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
            self.learning_rate = lambda f: f * self.learning_rate

        # Loggings
        self.vf_loss = 0.0
        self.entropy_loss = 0.0
        self.coverage_loss = 0.0

        self.ex_advantages_coef = 2
        self.in_advantages_coef = 1
        self.clip_normalization = 5

        if is_training_mode:
            self.actor.train()
            self.ex_critic.train()
            self.in_critic.train()
        else:
            self.actor.eval()
            self.ex_critic.eval()
            self.in_critic.eval()

    # def save_eps(self, state, action, reward, done, next_state, coverage_hist=1):
    #     self.memory.save_eps(state, action, reward, done, next_state, coverage_hist)

    # def save_observation(self, obs):
    #     self.obs_memory.save_eps(obs)

    # def update_obs_normalization_param(self, obs):
          
    #     obs = torch.FloatTensor(obs).to(self.device).detach()

    #     mean_obs = self.utils.count_new_mean(
    #         self.obs_memory.mean_obs, self.obs_memory.total_number_obs, obs)
    #     std_obs = self.utils.count_new_std(
    #         self.obs_memory.std_obs, self.obs_memory.total_number_obs, obs)
    #     total_number_obs = len(obs) + self.obs_memory.total_number_obs

    #     self.obs_memory.save_observation_normalize_parameter(
    #         mean_obs, std_obs, total_number_obs)

    # def update_rwd_normalization_param(self, in_rewards):
    #     std_in_rewards = self.utils.count_new_std(
    #         self.obs_memory.std_in_rewards, self.obs_memory.total_number_rwd, in_rewards)
    #     total_number_rwd = len(in_rewards) + self.obs_memory.total_number_rwd

    #     self.obs_memory.save_rewards_normalize_parameter(
    #         std_in_rewards, total_number_rwd)

    # Loss for RND
    def get_rnd_loss(self, state_pred, state_target):
        # Don't update target state value
        state_target = state_target.detach()

        # Mean Squared Error Calculation between state and predict
        forward_loss = ((state_target - state_pred).pow(2) * 0.5).mean()
        return forward_loss

    # Advantage for PPO

    def get_ADV_togo(self, memory_buffer, obs_memory_buffer, next_done):

       
        # Optimize policy for K epochs:
        states, _,ex_rewards, dones, next_states,_ = memory_buffer.fetch_all(numpy=False, to_device=self.device)
        mean_obs, std_obs, std_in_rewards = obs_memory_buffer.mean_obs.float(), obs_memory_buffer.std_obs.float(), obs_memory_buffer.std_in_rewards.float()

        obs = self.utils.normalize(next_states, mean_obs, std_obs, self.clip_normalization).detach()

        state_preds = self.rnd_predict(obs)
        state_targets = self.rnd_target(obs)
       
        ex_values, in_values = self.ex_critic(states),  self.in_critic(states)
    
        # old_ex_values = self.ex_critic_old(states)
        
        next_ex_values, next_in_values = self.ex_critic(next_states),  self.in_critic(next_states)
     
        # Don't use old value in backpropagation
        # Old_ex_values = old_ex_values.detach()

        # Computing internal reward, then getting internal general advantages estimator
        with torch.no_grad():
            in_rewards = (state_targets - state_preds).pow(2) * \
                0.5 / (std_in_rewards.mean() + 1e-8)
            in_rewards = in_rewards.squeeze(-1)
            # Getting external general advantages estimator
            ext_advantages = torch.zeros_like(ex_rewards).to(self.device)
            int_advantages = torch.zeros_like(in_rewards).to(self.device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(len(ex_rewards))):
                if t == len(ex_rewards) - 1:
                    ext_nextnonterminal = 1.0 - torch.Tensor(next_done).to(self.device)
                    int_nextnonterminal = 1.0
                    ext_nextvalues = next_ex_values[-1]
                    int_nextvalues = next_in_values[-1]
                else:
                    ext_nextnonterminal = 1.0 - dones[t+1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ex_values[t+1]
                    int_nextvalues = in_values[t+1]
                ext_delta = ex_rewards[t] + self.gamma * ext_nextvalues.reshape(1,-1)  * ext_nextnonterminal - ex_values[t].reshape(1,-1) 
                int_delta = in_rewards[t] + self.gamma * int_nextvalues.reshape(1,-1) * int_nextnonterminal - in_values[t].reshape(1,-1) 
                ext_advantages[t] = ext_lastgaelam = ext_delta + self.gamma * self.lam * ext_nextnonterminal * ext_lastgaelam
                int_advantages[t] = int_lastgaelam = int_delta + self.gamma * self.lam * int_nextnonterminal * int_lastgaelam
            External_Returns  = ext_advantages + ex_values.squeeze(-1)
            Internal_Returns = int_advantages + in_values.squeeze(-1)
        
            
            
            External_Advantages = self.utils.normalize(ext_advantages).detach()
            
            Internal_Advantages = self.utils.normalize(int_advantages).detach()

            # Getting overall advantages
            total_advantages = (self.ex_advantages_coef * External_Advantages +
                        self.in_advantages_coef * Internal_Advantages).detach()
        
        # External_Advantages = self.policy_function.generalized_advantage_estimation(ex_values, ex_rewards, next_ex_values, dones)
        # External_Returns = (External_Advantages + ex_values).detach()
        # External_Advantages = self.utils.normalize(External_Advantages).detach()

  
        # Internal_Advantages = self.policy_function.generalized_advantage_estimation(in_values, in_rewards, next_in_values, dones)
        # Internal_Returns = (Internal_Advantages + in_values).detach()
        # Internal_Advantages = self.utils.normalize(Internal_Advantages).detach()

        # # Getting overall advantages
        # total_advantages = (self.ex_advantages_coef * External_Advantages +
        #               self.in_advantages_coef * Internal_Advantages).detach()


        return total_advantages, External_Returns.detach(), Internal_Returns.detach()
        
    
    # Loss for PPO
    def get_PPO_loss_mb(self,states_mb, total_advantages, External_Returns, Internal_Returns, in_values, ex_values, Old_ex_values, action_probs, old_action_probs, actions):

        # # Don't use old value in backpropagation
        # Old_ex_values = old_ex_values.detach()

        # # Getting external general advantages estimator
        # External_Advantages = self.policy_function.generalized_advantage_estimation(
        #     ex_values, ex_rewards, next_ex_values, dones)
        # External_Returns = (External_Advantages + ex_values).detach()
        # External_Advantages = self.utils.normalize(
        #     External_Advantages).detach()

        # # Computing internal reward, then getting internal general advantages estimator
        # in_rewards = (state_targets - state_preds).pow(2) * \
        #     0.5 / (std_in_rewards.mean() + 1e-8)
        # Internal_Advantages = self.policy_function.generalized_advantage_estimation(
        #     in_values, in_rewards, next_in_values, dones)
        # Internal_Returns = (Internal_Advantages + in_values).detach()
        # Internal_Advantages = self.utils.normalize(
        #     Internal_Advantages).detach()

        # # Getting overall advantages
        # Advantages = (self.ex_advantages_coef * External_Advantages +
        #               self.in_advantages_coef * Internal_Advantages).detach()

        # Finding the ratio (pi_theta / pi_theta__old):
        #logprobs = self.distributions.logprob(action_probs, actions)
        norm_adv = False
        ex_values, in_values = self.ex_critic(states_mb),  self.in_critic(states_mb)
        if norm_adv:
            total_advantages = (total_advantages - total_advantages.mean()) / (total_advantages.std() + 1e-8)



        logprobs = self.distributions.logprob(action_probs, actions, mask=self.action_mask)
        #Old_logprobs = self.distributions.logprob(old_action_probs, actions).detach()
        Old_logprobs = self.distributions.logprob(old_action_probs, actions, mask=self.action_mask).detach()
        # ratios = old_logprobs / logprobs
        logratio = logprobs - Old_logprobs
        ratios = (logprobs - Old_logprobs).exp()

        # Finding KL Divergence
        Kl = self.distributions.kl_divergence(old_action_probs, action_probs, mask=self.action_mask)

        # with torch.no_grad():
        #     # calculate approx_kl http://joschu.net/blog/kl-approx.html
        #     # old_approx_kl = (-logratio).mean()
        #     Kl = ((ratios - 1) - logratio).mean()
        # Combining TR-PPO with Rollback (Truly PPO)
        pg_loss = torch.where(
            (Kl >= self.policy_kl_range) & (ratios > 1),
            ratios * total_advantages - self.policy_params * Kl,
            ratios * total_advantages
        )
        pg_loss = pg_loss.mean()

        # Getting entropy from the action probability
        dist_entropy = self.distributions.entropy(action_probs, mask=self.action_mask).mean()
        self.entropy_loss = dist_entropy
        # Getting critic loss by using Clipped critic value
        # Minimize the difference between old value and new value
        ex_vpredclipped = Old_ex_values + \
            torch.clamp(ex_values - Old_ex_values, -
                        self.value_clip, self.value_clip)
        # Mean Squared Error
        ex_vf_losses1 = (External_Returns - ex_values.squeeze(-1)).pow(2)
        # Mean Squared Error
        ex_vf_losses2 = (External_Returns - ex_vpredclipped.squeeze(-1)).pow(2)
        critic_ext_loss = torch.max(ex_vf_losses1, ex_vf_losses2).mean()
        #self.vf_loss = critic_ext_loss
        # Getting Intrinsic critic loss
        critic_int_loss = (Internal_Returns - in_values.squeeze(-1)).pow(2).mean()
        #critic_int_loss = 0.5 *((in_values.squeeze(-1) - Internal_Returns ) ** 2).mean()

        # Getting overall critic loss
        critic_loss = (critic_ext_loss + 0.5 * critic_int_loss) * 0.5
        #critic_loss = critic_ext_loss 
        self.vf_loss = critic_loss
        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss
        #loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss 
        return pg_loss, critic_loss, dist_entropy

    def act(self, state, action_mask_pro=None, coverage_hist=None):
        if not torch.is_tensor(state):
      
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(self.device).detach()
        
        if coverage_hist:
            coverage_hist = torch.FloatTensor(coverage_hist).to(self.device).detach()

            with torch.no_grad():
                action_probs = self.actor(state, coverage_hist)#.cpu()
        else:
            with torch.no_grad():
                action_probs = self.actor(state)#.cpu()

        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        
        if self.is_training_mode:
            # Sample the action
            
            if self.mask_type == "Soft":
                action = self.distributions.sample(action_probs, mask=self.action_mask)
            elif self.mask_type == "Hard":
                action_mask_pro_tensor = torch.tensor(action_mask_pro, device=self.device).float()
                action = self.distributions.sample(action_probs*action_mask_pro_tensor)
            else:
                action = self.distributions.sample(action_probs)
        else:
            action = torch.argmax(action_probs, 1)

        return action#.cpu().item()

    def compute_intrinsic_reward(self, obs, mean_obs, std_obs):
        obs = self.utils.normalize(obs, mean_obs, std_obs).detach()
        with torch.no_grad():
            state_pred = self.rnd_predict(obs)
            state_target = self.rnd_target(obs)

        return (state_target - state_pred)

    # Get loss and Do backpropagation
    def training_rnd(self, obs, obs_memory_buffer):
        mean_obs, std_obs = obs_memory_buffer.mean_obs.float(), obs_memory_buffer.std_obs.float()
        obs = self.utils.normalize(obs, mean_obs, std_obs).detach()

        state_pred = self.rnd_predict(obs)
        state_target = self.rnd_target(obs)
        loss = self.get_rnd_loss(state_pred, state_target)
        self.rnd_predict_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.rnd_predict.parameters(), 0.5)
        self.rnd_predict_optimizer.step()
        

    # Get loss and Do backpropagation
    def training_ppo_mb(self, states_mb, actions_mb, coverage_hist, total_advantages_mb, External_Returns_mb, Internal_Returns_mb):
        old_ex_values = self.ex_critic_old(states_mb)
        Old_ex_values = old_ex_values.detach()
           
        # Don't update rnd value
        
        action_probs, ex_values, in_values = self.actor(states_mb, coverage_hist), self.ex_critic(states_mb),  self.in_critic(states_mb)
    
        old_action_probs, old_ex_values = self.actor_old(states_mb, coverage_hist), self.ex_critic_old(states_mb)
        
    
        if self.coverage:
            #loss = self.get_PPO_loss(action_probs, ex_values, old_action_probs, old_ex_values, next_ex_values, actions, rewards, dones,
            #                        state_preds, state_targets, in_values, old_in_values, next_in_values, std_in_rewards)

            coverage_loss = torch.mean(torch.min(coverage_hist, torch.nn.functional.softmax(action_probs, 0)))

            self.coverage_loss = coverage_loss

            if self.coverage_coef is None:
                loss = loss + self.entropy_loss * coverage_loss
            else:
                loss = loss + self.coverage_coef * coverage_loss
            
        else:
            pg_loss, critic_loss, dist_entropy = self.get_PPO_loss_mb(states_mb, total_advantages_mb, External_Returns_mb, Internal_Returns_mb, \
                in_values, ex_values, Old_ex_values, action_probs, old_action_probs, actions_mb)
            self.coverage_loss = 0                        

        self.actor_optimizer.zero_grad()
        self.ex_critic_optimizer.zero_grad()
        self.in_critic_optimizer.zero_grad()

        loss = - (dist_entropy * self.entropy_coef) - pg_loss + (critic_loss * self.vf_loss_coef) 
        
        loss.backward()
        # max_grad_norm = 0.5
        # nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.ex_critic.parameters()), max_grad_norm)

        self.actor_optimizer.step()
        self.ex_critic_optimizer.step()
        self.in_critic_optimizer.step()

        return pg_loss, critic_loss, dist_entropy

    # Update the model
    def update_rnd(self, rollout_runner):
        batch_size = int(len(rollout_runner.obs_buffer) / self.minibatch)
        #batch_size = self.minibatch
        dataloader = DataLoader(rollout_runner.obs_buffer, batch_size, shuffle=False)
        #K_epoch = int(np.ceil(len(self.obs_memory)*self.RND_epochs/self.minibatch))
        # Optimize policy for K epochs:
        for _ in range(self.RND_epochs):
            for obs in dataloader:
                self.training_rnd(obs.float().to(self.device), rollout_runner.obs_buffer)
        
        intrinsic_rewards = self.compute_intrinsic_reward(rollout_runner.obs_buffer.get_all().to(
            self.device), rollout_runner.obs_buffer.mean_obs.to(self.device), rollout_runner.obs_buffer.std_obs.to(self.device))
        
        rollout_runner.update_obs_normalization_param(rollout_runner.obs_buffer.observations)
        rollout_runner.update_rwd_normalization_param(intrinsic_rewards)
       
        # Clear the memory
        rollout_runner.obs_buffer.clear_memory()
        return rollout_runner
    # Update the model
    def update_ppo(self, rollout_runner, next_done_flag, writer=None):

        """
        speperate the whole process into two

        """


        total_advantages, External_Returns, Internal_Returns = self.get_ADV_togo(rollout_runner.rollout_buffer, rollout_runner.obs_buffer, next_done_flag)


        rollout_runner.adv_buffer = TensorDataset(total_advantages.to(self.device), External_Returns, Internal_Returns) # create your datset
        


        mini_batch_size = len(rollout_runner.rollout_buffer) // self.minibatch
        #print(f"len_memory = {len(rollout_runner.rollout_buffer)}")
        #print(f"mini_batch_size = {mini_batch_size}")
        #batch_size = self.minibatch
        data_loader = DataLoader(rollout_runner.rollout_buffer, mini_batch_size, shuffle=False)
        adv_loader = DataLoader(rollout_runner.adv_buffer, mini_batch_size, shuffle=False)
        # Optimize policy for K epochs:
        #K_epoch = int(np.ceil(self.minibatch*self.PPO_epochs))
        #print(f"K_epoch = {K_epoch}")
        for i in range(self.PPO_epochs):
            #print(i)
            for mini_batch_itre, data in enumerate(zip(cycle(data_loader), adv_loader)):

            #for states, actions, rewards, dones, next_states, coverage_hist in dataloader:
                #print(mini_batch_itre)
                #print(data)

                states_mb, actions_mb, rewards_mb, dones_mb, next_states_mb, coverage_hist_mb = data[0]
                total_advantages_mb, External_Returns_mb, Internal_Returns_mb = data[1]

                pg_loss, critic_loss, dist_entropy = self.training_ppo_mb(states_mb.to(self.device), actions_mb.to(self.device),coverage_hist_mb.to(self.device),\
                                     total_advantages_mb, External_Returns_mb, Internal_Returns_mb)
        
        # Clear the memory
        rollout_runner.rollout_buffer.clear_memory()

        # Copy new weights into old policy:
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.ex_critic_old.load_state_dict(self.ex_critic.state_dict())
        self.in_critic_old.load_state_dict(self.in_critic.state_dict())
        if not writer:
            return rollout_runner
        else:
            return rollout_runner, pg_loss, critic_loss, dist_entropy

    def save_weights(self):
        torch.save({
            'model_state_dict': self.actor.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict(),
        }, './test/My Drive/PPO_agent/actor.tar')

        torch.save({
            'model_state_dict': self.ex_critic.state_dict(),
            'optimizer_state_dict': self.ex_critic_optimizer.state_dict()
        }, './test/My Drive/PPO_agent/ex_critic.tar')

        torch.save({
            'model_state_dict': self.in_critic.state_dict(),
            'optimizer_state_dict': self.in_critic_optimizer.state_dict()
        }, './test/My Drive/PPO_agent/in_critic.tar')

    def load_weights(self):
        actor_checkpoint = torch.load('./test/My Drive/PPO_agent/actor.tar')
        self.actor.load_state_dict(actor_checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(
            actor_checkpoint['optimizer_state_dict'])

        ex_critic_checkpoint = torch.load(
            './test/My Drive/PPO_agent/ex_critic.tar')
        self.ex_critic.load_state_dict(
            ex_critic_checkpoint['model_state_dict'])
        self.ex_critic_optimizer.load_state_dict(
            ex_critic_checkpoint['optimizer_state_dict'])

        in_critic_checkpoint = torch.load(
            './test/My Drive/PPO_agent/in_critic.tar')
        self.in_critic.load_state_dict(
            in_critic_checkpoint['model_state_dict'])
        self.in_critic_optimizer.load_state_dict(
            in_critic_checkpoint['optimizer_state_dict'])
