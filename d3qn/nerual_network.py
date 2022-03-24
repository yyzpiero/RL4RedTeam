from torch import nn
from typing import List, Dict, Iterable, Optional, Union
from torch.nn import Parameter, init
import torch
import math
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.container import Sequential


class FilterLegalMoves(nn.Module):
    """
    Custom layer to consider only valid moves
    """

    def forward(self, x, possible_moves):
        legal_moves = possible_moves
        actions_tensor = torch.zeros(x.shape).to(x.device)
        batch_size = x.shape[0]
        for i in range(batch_size):
            actions_tensor[i, legal_moves[i]] = 1.0
        filtered_actions = x * actions_tensor
        filtered_actions[filtered_actions == 0] = -np.inf
        return filtered_actions


class NoisyLinear(nn.Module):
    def __init__(self, in_dim, out_dim, params=None, is_noisy=False):
        super(NoisyLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.params = params
        self.is_noisy = is_noisy
        self.sigma_init = 0.4
        self.mu_w = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.mu_b = nn.Parameter(torch.Tensor(out_dim))
        self.sigma_w = nn.Parameter(torch.Tensor(out_dim, in_dim))
        self.sigma_b = nn.Parameter(torch.Tensor(out_dim))
        # Epsilon is not trainable
        self.register_buffer("eps_w", torch.Tensor(out_dim, in_dim))
        self.register_buffer("eps_b", torch.Tensor(out_dim))
        self.init_params()
        self.reset_noise()

    def init_params(self):
        # Trainable params
        nn.init.uniform_(
            self.mu_w, -math.sqrt(1 / self.in_dim), math.sqrt(1 / self.in_dim)
        )
        nn.init.uniform_(
            self.mu_b, -math.sqrt(1 / self.in_dim), math.sqrt(1 / self.in_dim)
        )
        nn.init.constant_(self.sigma_w, self.sigma_init / math.sqrt(self.out_dim))
        nn.init.constant_(self.sigma_b, self.sigma_init / math.sqrt(self.out_dim))

    def reset_noise(self):
        self.eps_w.copy_(
            self.factorize_noise(self.out_dim).ger(self.factorize_noise(self.in_dim))
        )
        self.eps_b.copy_(self.factorize_noise(self.out_dim))

    def factorize_noise(self, size):
        # Modify scale to amplify or reduce noise
        x = torch.Tensor(np.random.normal(loc=0.0, scale=0.001, size=size))
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x):
        return F.linear(
            x,
            self.mu_w + self.sigma_w * self.eps_w,
            self.mu_b + self.sigma_b * self.eps_b,
        )


# class NoisyLinear(nn.Module):
#     def __init__(self, in_features, out_features, std_init=0.4):
#         super(NoisyLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         # Uniform Distribution bounds:
#         #     U(-1/sqrt(p), 1/sqrt(p))
#         self.lowerU = -1.0 / math.sqrt(in_features)  #
#         self.upperU = 1.0 / math.sqrt(in_features)  #
#         self.sigma_0 = std_init
#         self.sigma_ij_in = self.sigma_0 / math.sqrt(self.in_features)
#         self.sigma_ij_out = self.sigma_0 / math.sqrt(self.out_features)

#         """
#         Registre_Buffer: Adds a persistent buffer to the module.
#             A buffer that is not to be considered as a model parameter -- like "running_mean" in BatchNorm
#             It is a "persistent state" and can be accessed as attributes --> self.weight_epsilon
#         """
#         self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
#         self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
#         self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

#         self.bias_mu = nn.Parameter(torch.empty(out_features))
#         self.bias_sigma = nn.Parameter(torch.empty(out_features))
#         self.register_buffer("bias_epsilon", torch.empty(out_features))

#         self.reset_parameters()
#         self.reset_noise()

#     def reset_parameters(self):
#         self.weight_mu.data.uniform_(self.lowerU, self.upperU)
#         self.weight_sigma.data.fill_(self.sigma_ij_in)

#         self.bias_mu.data.uniform_(self.lowerU, self.upperU)
#         self.bias_sigma.data.fill_(self.sigma_ij_out)

#     def reset_noise(self):
#         eps_in = self.func_f(self.in_features)
#         eps_out = self.func_f(self.out_features)
#         # Take the outter product
#         """
#             >>> v1 = torch.arange(1., 5.) [1, 2, 3, 4]
#             >>> v2 = torch.arange(1., 4.) [1, 2, 3]
#             >>> torch.ger(v1, v2)
#             tensor([[  1.,   2.,   3.],
#                     [  2.,   4.,   6.],
#                     [  3.,   6.,   9.],
#                     [  4.,   8.,  12.]])
#         """
#         # outer product
#         self.weight_epsilon.copy_(eps_out.ger(eps_in))
#         self.bias_epsilon.copy_(eps_out)

#     def func_f(self, n):  # size
#         # sign(x) * sqrt(|x|) as in paper
#         x = torch.rand(n)
#         return x.sign().mul_(x.abs().sqrt_())

#     def forward(self, x):
#         if self.training:
#             return F.linear(
#                 x,
#                 self.weight_mu + self.weight_sigma * self.weight_epsilon,
#                 self.bias_mu + self.bias_sigma * self.bias_epsilon,
#             )

#         else:
#             return F.linear(x, self.weight_mu, self.bias_mu)


class DuelingDQN(nn.Module):
    """
	Dueling DQN -> http://proceedings.mlr.press/v48/wangf16.pdf
	"""

    def __init__(self, input_shape, n_actions):
        super(DuelingDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        # Predict the actions advantage
        self.fc_a = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

        # Predict the state value
        self.fc_v = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))  # apply convolution layers..
        return int(np.prod(o.size()))  # ..to obtain the output shape

    def forward(self, x):
        batch_size = x.size()[0]
        conv_out = self.conv(x).view(
            batch_size, -1
        )  # apply convolution layers and flatten the results

        adv = self.fc_a(conv_out)
        val = self.fc_v(conv_out)

        # Sum the state value with the advantage of each action (NB: the mean has been subtracted from the advantage. It is used in the paper)
        return val + adv - torch.mean(adv, dim=1, keepdim=True)


class DQNBuild(nn.Module):
    def __init__(
        self,
        input_dim,
        n_actions: int,
        net_arch: List[int],
        noisy_net: bool,
        dueling: bool,
        c51: bool,
        Vmin,
        Vmax,
        num_atoms,
        batch_size,
    ):
        super(DQNBuild, self).__init__()
        activation_fn = nn.ReLU
        self.noisy_net = noisy_net
        self.dueling = dueling
        self.c51 = c51
        self.num_actions = n_actions
        self.num_atoms = num_atoms
        # print(self.c51)
        # print(self.noisy_net)
        # print(self.dueling)

        if len(net_arch) > 0:
            modules = [
                nn.Linear(input_dim[0], net_arch[0]),
                # nn.BatchNorm1d(net_arch[0]),
                activation_fn(),
            ]
        else:
            modules = []

        if self.noisy_net:
            for idx in range(len(net_arch) - 1):
                modules.append(NoisyLinear(net_arch[idx], net_arch[idx + 1]))
                # modules.append(nn.BatchNorm1d(net_arch[idx + 1]))
                modules.append(activation_fn())
            if self.dueling:
                if self.c51:

                    support = torch.linspace(Vmin, Vmax, self.num_atoms)
                    offset = (
                        torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size)
                        .long()
                        .unsqueeze(1)
                        .expand(batch_size, self.num_atoms)
                    )

                    self.register_buffer("support", support)
                    self.register_buffer("offset", offset)

                    self.q_net_common = nn.Sequential(*modules)
                    self.advantage = nn.Sequential(
                        NoisyLinear(net_arch[-1], 128),
                        NoisyLinear(128, self.num_actions * self.num_atoms),
                    )
                    self.value = nn.Sequential(
                        NoisyLinear(net_arch[-1], 128),
                        NoisyLinear(128, self.num_atoms),
                    )
                    self.softmax = nn.Softmax(dim=1)

                else:
                    self.q_net_common = nn.Sequential(*modules)
                    self.advantage = nn.Sequential(
                        NoisyLinear(net_arch[-1], 128),
                        NoisyLinear(128, self.num_actions),
                    )
                    self.value = nn.Sequential(
                        NoisyLinear(net_arch[-1], 128), NoisyLinear(128, 1),
                    )
                    self.custom_softmax = FilterLegalMoves()
            else:
                if self.num_actions > 0:
                    last_linear_head_dim = (
                        net_arch[-1] if len(net_arch) > 0 else input_dim[0]
                    )
                    modules.append(NoisyLinear(last_linear_head_dim, self.num_actions))
                self.q_net = nn.Sequential(*modules)

        else:
            for idx in range(len(net_arch) - 1):
                modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
                # modules.append(nn.BatchNorm1d(net_arch[idx + 1]))
                modules.append(activation_fn())
            if self.c51:
                if self.dueling:
                    support = torch.linspace(Vmin, Vmax, self.num_atoms)
                    offset = (
                        torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size)
                        .long()
                        .unsqueeze(1)
                        .expand(batch_size, self.num_atoms)
                    )

                    self.register_buffer("support", support)
                    self.register_buffer("offset", offset)

                    self.q_net_common = nn.Sequential(*modules)
                    self.advantage = nn.Sequential(
                        nn.Linear(net_arch[-1], 128),
                        activation_fn(),
                        nn.Linear(128, self.num_actions * self.num_atoms),
                    )
                    self.value = nn.Sequential(
                        nn.Linear(net_arch[-1], 128),
                        activation_fn(),
                        nn.Linear(128, self.num_atoms),
                    )
                    self.softmax = nn.Softmax(dim=1)

            else:
                if self.dueling:
                    self.q_net_common = nn.Sequential(*modules)
                    self.advantage = nn.Sequential(
                        nn.Linear(net_arch[-1], 128),
                        activation_fn(),
                        nn.Linear(128, self.num_actions),
                    )
                    self.value = nn.Sequential(
                        nn.Linear(net_arch[-1], 128), activation_fn(), nn.Linear(128, 1)
                    )

                else:
                    if self.num_actions > 0:
                        last_linear_head_dim = (
                            net_arch[-1] if len(net_arch) > 0 else input_dim[0]
                        )
                        modules.append(
                            nn.Linear(last_linear_head_dim, self.num_actions)
                        )
                    self.q_net = nn.Sequential(*modules)
        # self.q_net = nn.Sequential(*modules)

        # if noisy_net:

        # self.head = nn.Linear(linear_input_size, n_actions)

    def forward(self, features, Dueling):
        # TODO: detach the linear head
        # TODO: seperate policy network and values network
        if self.dueling:
            if self.c51:
                x = self.q_net_common(features)
                advantage = self.advantage(x).view(-1, self.num_actions, self.num_atoms)
                value = self.value(x).view(-1, 1, self.num_atoms)

                x = value + advantage - advantage.mean(1, keepdim=True)
                x = self.softmax(x.view(-1, self.num_atoms))
                x = x.view(-1, self.num_actions, self.num_atoms)
                return x
            else:
                x = self.q_net_common(features)
                advantage, value = self.advantage(x), self.value(x)
                return value + (advantage - advantage.mean(1, keepdim=True))

        else:
            return self.q_net(features)

    def update_noisy_modules(self):
        if self.noisy_net:
            self.noisy_modules = [
                module for module in self.modules() if isinstance(module, NoisyLinear)
            ]

    def reset_noise(self):
        self.noisy_modules = [
            module for module in self.modules() if isinstance(module, NoisyLinear)
        ]
        for module in self.noisy_modules:
            module.reset_noise()

    def remove_noise(self):
        for module in self.noisy_modules:
            module.remove_noise()


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.feature = NoisyLinear(in_dim[0], 128)
        self.noisy_layer1 = NoisyLinear(128, 128)
        self.noisy_layer2 = NoisyLinear(128, out_dim)

        # self.q_net = Sequential(self.feature, self.noisy_layer1, self.noisy_layer2)

    def forward(self, x: torch.Tensor, dueling: bool) -> torch.Tensor:
        """Forward method implementation."""
        feature = F.relu(self.feature(x))
        hidden = F.relu(self.noisy_layer1(feature))
        out = self.noisy_layer2(hidden)
        # out = self.q_net(x)
        return out

    def reset_noise(self):
        """Reset all noisy layers."""
        self.feature.reset_noise()
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()

    # def reset_noise(self):
    #     self.noisy_modules = [
    #         module for module in self.modules() if isinstance(module, NoisyLinear)
    #     ]
    #     for module in self.noisy_modules:
    #         module.reset_noise()


"""
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
"""

