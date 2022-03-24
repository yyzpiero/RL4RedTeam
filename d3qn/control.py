import torch as torch
import torch.nn as nn
import numpy as np
from nerual_network import DQNBuild, DuelingDQN, Network
from my_utils import polyak_update

"""
In charge of control the training and prediction process
update targets
deteriming training prcesss
call NN builders
calculate loss -- optimizers

"""


class Controller:
    def __init__(
        self,
        obs_space_shape,
        action_space_shape,
        gamma,
        net_arch,
        batch_size,
        c51,
        dueling,
        noisy_net,
        double_DQN,
        n_multi_step,
        Vmax,
        Vmin,
        num_atoms,
        device,
    ):
        # net_arch = [64]

        self.gamma = gamma
        self.double_DQN = double_DQN
        self.c51 = c51
        self.noisy_net = noisy_net
        self.dueling = dueling
        self.n_multi_step = n_multi_step
        self.batch_size = batch_size
        self.device = device
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.num_atoms = num_atoms

        # print(self.c51)
        # print(self.noisy_net)
        # print(self.dueling)

        self.target_q_nn = DQNBuild(
            obs_space_shape,
            action_space_shape,
            net_arch,
            noisy_net,
            dueling,
            c51,
            Vmin=self.Vmin,
            Vmax=self.Vmax,
            num_atoms=self.num_atoms,
            batch_size=self.batch_size,
        ).to(device)
        self.moving_q_nn = DQNBuild(
            obs_space_shape,
            action_space_shape,
            net_arch,
            noisy_net,
            dueling,
            c51,
            Vmin=self.Vmin,
            Vmax=self.Vmax,
            num_atoms=self.num_atoms,
            batch_size=self.batch_size,
        ).to(device)

        # if self.noisy_net:
        #     self.target_q_nn = Network(obs_space_shape, action_space_shape).to(
        #         self.device
        #     )
        #     self.moving_q_nn = Network(obs_space_shape, action_space_shape).to(
        #         self.device
        #     )

    def get_max_action(self, obs, dueling):
        """
        Forward pass of the NN to obtain the action of the given observations
        """
        # if self.noisy_net:
        #     self.target_q_nn.sample_noise()
        #     self.moving_q_nn.sample_noise()

        if self.c51:
            with torch.no_grad():
                state = torch.tensor(np.array([obs])).to(self.device)
                self.moving_q_nn.eval()
                q_dist = self.moving_q_nn.forward(state.float(), dueling)
                q_value = (q_dist * self.moving_q_nn.support).sum(2)
                action = q_value.max(1)[1].item()
        else:
            # convert the observation in tensor
            state_t = torch.tensor(np.array([obs])).to(self.device)
            # forawrd pass
            self.moving_q_nn.eval()
            with torch.no_grad():
                q_values_t = self.moving_q_nn(state_t.float(), dueling)
            # self.moving_q_nn.train()
            # get the maximum value of the output (i.e. the best action to take)
            _, act_t = torch.max(q_values_t, dim=1)
            action = int(act_t.item())

        return action

    def optimizer_build(self, learning_rate, optim_type="Adam", gamma=0.9):

        # def set_optimizer(self, learning_rate):
        if optim_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.moving_q_nn.parameters(), lr=learning_rate
            )
        elif optim_type == "SGD":
            self.optimizer = torch.optim.SGD(
                self.moving_q_nn.parameters(), lr=learning_rate
            )
        elif optim_type == "RMSProp":
            self.optimizer = torch.optim.RMSprop(
                self.moving_q_nn.parameters(), lr=learning_rate
            )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.99
        )

    def optimize(self, mini_batch, batch_size):
        """
        Optimize the NN
        """
        # reset the grads
        self.optimizer.zero_grad()
        # caluclate the loss of the mini batch
        loss = self._calulate_loss(mini_batch, batch_size)
        loss_v = loss.item()

        # do backpropagation
        loss.backward()
        # for param in list(list(self.moving_q_nn.parameters())):
        #     param.grad.data.clamp_(-1, 1)
        # one step of optimization
        self.optimizer.step()
        # self.scheduler.step()
        if self.noisy_net:
            self.moving_q_nn.reset_noise()
            self.target_q_nn.reset_noise()
        return loss_v

    def update_target(self):
        self.target_q_nn.load_state_dict(self.moving_q_nn.state_dict())
        #polyak_update(self.moving_q_nn.parameters(), self.target_q_nn.parameters())

    def _calulate_loss(self, mini_batch, batch_size):
        """
        Calculate mini batch's MSE loss.
        It support also the double DQN version
        """

        states, actions, next_states, rewards, dones = mini_batch
        weights = torch.ones(batch_size)
        # convert the data in tensors
        states_t = torch.as_tensor(states, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        done_t = torch.as_tensor(dones, dtype=torch.uint8, device=self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Value of the action taken previously (recorded in actions_v) in the state_t
        if not self.c51:
            state_action_values = (
                self.moving_q_nn(states_t, self.dueling)
                .gather(1, actions_t[:, None])
                .squeeze(-1)
            )
            # NB gather is a differentiable function

            # Next state value with Double DQN. (i.e. get the value predicted by the target nn, of the best action predicted by the moving nn)
            if self.double_DQN:
                double_max_action = self.moving_q_nn(next_states_t, self.dueling).max(
                    1
                )[1]
                double_max_action = double_max_action.detach()
                target_output = self.target_q_nn(next_states_t, self.dueling)
                next_state_values = torch.gather(
                    target_output, 1, double_max_action[:, None]
                ).squeeze(
                    -1
                )  # NB: [:,None] add an extra dimension

            # Next state value in the normal configuration
            else:
                next_state_values = self.target_q_nn(next_states_t, self.dueling).max(
                    1
                )[0]

            next_state_values = next_state_values.detach()  # No backprop

            # Use the Bellman equation
            expected_state_action_values = (
                rewards_t + (self.gamma ** self.n_multi_step) * next_state_values
            )
            # compute the loss
            # loss = nn.MSELoss()(state_action_values, expected_state_action_values)
            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            # criterion = nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values)
            loss = (loss * weights).mean()

        else:
            q_dist = self.moving_q_nn(states_t, self.dueling)
            actions_t = (
                actions_t.unsqueeze(1)
                .unsqueeze(1)
                .expand(batch_size, 1, self.num_atoms)
            )
            q_dist = q_dist.gather(1, actions_t).squeeze(1)
            q_dist.data.clamp_(0.01, 0.99)

            target_dist = self._projection_distribution(
                current_model=self.moving_q_nn,
                target_model=self.target_q_nn,
                next_state=next_states_t,
                reward=rewards_t,
                done=done_t,
                target_model_support=self.target_q_nn.support,
                target_model_offset=self.target_q_nn.offset,
            )
            loss = -(target_dist * q_dist.log()).sum(1)
            loss = (loss * weights).mean()

        return loss

    def _projection_distribution(
        self,
        current_model,
        target_model,
        next_state,
        reward,
        done,
        target_model_support,
        target_model_offset,
    ):
        delta_z = float(self.Vmax - self.Vmin) / (self.num_atoms - 1)

        target_next_q_dist = target_model(next_state, self.dueling)

        if self.double_DQN:
            next_q_dist = current_model(next_state, self.dueling)
            next_action = (next_q_dist * target_model_support).sum(2).max(1)[1]
        else:
            next_action = (target_next_q_dist * target_model_support).sum(2).max(1)[1]

        next_action = (
            next_action.unsqueeze(1)
            .unsqueeze(1)
            .expand(target_next_q_dist.size(0), 1, target_next_q_dist.size(2))
        )
        target_next_q_dist = target_next_q_dist.gather(1, next_action).squeeze(1)

        reward = reward.unsqueeze(1).expand_as(target_next_q_dist)
        done = done.unsqueeze(1).expand_as(target_next_q_dist)
        support = target_model_support.unsqueeze(0).expand_as(target_next_q_dist)

        Tz = reward + self.gamma * support * (1 - done)
        Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
        b = (Tz - self.Vmin) / delta_z
        l = b.floor().long()
        u = b.ceil().long()

        target_dist = target_next_q_dist.clone().zero_()
        target_dist.view(-1).index_add_(
            0,
            (l + target_model_offset).view(-1),
            (target_next_q_dist * (u.float() - b)).view(-1),
        )
        target_dist.view(-1).index_add_(
            0,
            (u + target_model_offset).view(-1),
            (target_next_q_dist * (b - l.float())).view(-1),
        )

        return target_dist
