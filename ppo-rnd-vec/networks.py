import numpy as np
import torch
import torch.nn as nn

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Linear_Coverage_Model(nn.Module):
    def __init__(self, input_dim, out_dim, device='cuda'):
        super(Linear_Coverage_Model, self).__init__()

        self.linear_coverage = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        ).float().to(device)
        #nn.Linear(input_dim, out_dim,bias=True).to(device)
        #self.linear = nn.Linear(input_dim, out_dim,bias=True).to(device)
    def forward(self, C):
        out =  self.linear_coverage(C)   #+ self.linear(X)     
        return out

class Actor_Model_Coverage(nn.Module):
    def __init__(self, state_dim, action_dim, num_hidden=256, mask_type=None, device='cuda'):
        super(Actor_Model_Coverage, self).__init__()
        
        self.mask_type = mask_type
        self.nn_layer = nn.Sequential(
            layer_init(nn.Linear(state_dim[0], num_hidden), std=0.01),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, action_dim)
        ).float().to(device)

        self.cw_learner = nn.Sequential(nn.Linear(state_dim[0], action_dim),nn.Tanh()).float().to(device)
        self.aw_learner = nn.Sequential(nn.Linear(state_dim[0], action_dim),nn.Tanh()).float().to(device)

        self.linear_coverage = Linear_Coverage_Model(action_dim, action_dim).float().to(device)
        self.cout_activation = nn.Tanh().float().to(device)
        self.out_softmax = nn.Softmax(-1).float().to(device)

    def forward(self, states, coverage_hist):
        fusion = True
        if fusion:
            out = self.nn_layer(states)

            c_w = self.cw_learner(states)
            a_w = self.aw_learner(states)

            c_out = self.linear_coverage(coverage_hist)
            c_out = self.cout_activation(c_out)
            #c_out = a_w * c_out

            if self.mask_type == "Soft":
                return a_w*out + c_w*c_out
            else:
                return self.out_softmax(a_w*out + c_w*c_out)
        else:
            out = self.nn_layer(states)
            
            c_out = self.linear_coverage(coverage_hist)
            c_out = self.cout_activation(c_out)
            
            if self.mask_type == "Soft":
                return out + c_out
            else:
                return self.out_softmax(out + c_out)
        #ut = self.out_activation2(out+c_out)

        
        #return out + c_out




class Actor_Model(nn.Module):
    def __init__(self, state_dim, action_dim, mask_type, num_hidden=256,  device='cuda'):
        super(Actor_Model, self).__init__()
        
        if mask_type == "Soft":
            self.nn_layer = nn.Sequential(
                layer_init(nn.Linear(state_dim[0], 256), std=0.01),
                nn.ReLU(),
                layer_init(nn.Linear(256, 256), std=0.01),
                nn.ReLU(),
                layer_init(nn.Linear(256, action_dim), std=0.01)
            ).float().to(device)
        else:
            self.nn_layer = nn.Sequential(
            layer_init(nn.Linear(state_dim[0], 256), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
            nn.Softmax(-1)
            ).float().to(device)

    def forward(self, states, coverage_hist=None):
        out = self.nn_layer(states)
        return out 

class Critic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, num_hidden=256, device='cuda'):
        super(Critic_Model, self).__init__()

        self.nn_layer = nn.Sequential(
            layer_init(nn.Linear(state_dim[0], num_hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(num_hidden, num_hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(num_hidden, 1))
        ).float().to(device)

    def forward(self, states):
        return self.nn_layer(states)


class RND_Model(nn.Module):
    def __init__(self, state_dim, action_dim, num_hidden=256, device='cuda'):
        super(RND_Model, self).__init__()

        self.nn_layer = nn.Sequential(
            nn.Linear(state_dim[0], num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        ).float().to(device)
        
    def forward(self, states):
        return self.nn_layer(states)
