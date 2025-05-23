import torch
import torch.nn as nn
import torch.nn.functional as F

activation = nn.LeakyReLU

class RewardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super(RewardNet, self).__init__()
        self.input_dim = input_dim
        last_dim = self.input_dim
        layer_list = []
        for i in range(num_layers):
            layer_list.append(nn.Linear(last_dim, hidden_dim))
            layer_list.append(activation())
            last_dim = hidden_dim
        layer_list.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layer_list)


    def forward(self, x):
        return self.net(x)


    def compute_reward(self, x):
        with torch.no_grad():
            x = torch.Tensor(x).float()
            if len(x.size()) == 1:
                x = x.view(1, -1)
            reward = self.net(x)
        return reward.item()   

class RewardNets(nn.Module):
    def __init__(self, input_dim, net_num, hidden_dim=256, num_layers=2):
        super(RewardNets, self).__init__()
        self.reward_nets = nn.ModuleList([RewardNet(input_dim, hidden_dim, num_layers) for i in range(net_num)])
        self.net_num = net_num

    def forward(self, x_list):
        if type(x_list) == list and len(x_list) == self.net_num:
            return [self.reward_nets[i](x_list[i]) for i in range(self.net_num)]
        else:
            return [self.reward_nets[i](x_list) for i in range(self.net_num)]

    def compute_reward(self, x):
        with torch.no_grad():
            x = torch.Tensor(x).float()
            if len(x.size()) == 1:
                x = x.view(1, -1)
            reward = np.sum([self.reward_nets[i](x).item() for i in range(self.net_num)])
        return reward

class GaussianRewardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2):
        super(GaussianRewardNet, self).__init__()
        self.input_dim = input_dim
        last_dim = input_dim
        layer_list = []
        for _ in range(num_layers):
            layer_list.append(nn.Linear(last_dim, hidden_dim))
            layer_list.append(activation())
            last_dim = hidden_dim
        # Output both mean and log variance
        layer_list.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        """
        Returns:
            mu:    (batch_size, 1)
            sigma: (batch_size, 1), std dev (positive)
        """
        out = self.net(x)
        mu = out[:, 0:1]
        log_var = out[:, 1:2]
        sigma = torch.nn.functional.softplus(log_var)  # ensure positive std
        return mu, sigma

    def compute_reward(self, x, return_std=False):
        """
        For evaluation. Returns mean reward and std dev as floats.
        """
        with torch.no_grad():
            x = torch.Tensor(x).float()
            if len(x.shape) == 1:
                x = x.view(1, -1)
            mu, sigma = self.forward(x)
        if return_std:
            return mu.item(), sigma.item()
        else:
            return mu.item()