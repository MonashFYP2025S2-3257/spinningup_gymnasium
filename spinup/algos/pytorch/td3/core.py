import numpy as np
import scipy.signal

import torch
import torch.nn as nn


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

class ResidualBlock(nn.Module):
    """
    A single layer that 'skips' over itself.
    It calculates: Activation(Linear(x) + x)
    """
    def __init__(self, size, activation):
        super().__init__()
        self.linear = nn.Linear(size, size)
        self.act = activation()

    def forward(self, x):
        # The 'skip': add the original input x back to the output of the layer
        return self.act(self.linear(x) + x)
    
class SkipMLP(nn.Module):
    """
    An MLP that uses ResidualBlocks for its hidden layers.
    """
    def __init__(self, sizes, activation):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(sizes[0], sizes[1]),
            activation()
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(sizes[1], activation) for _ in range(len(sizes) - 2)
        ])

        self.output_layer = nn.Linear(sizes[-2], sizes[-1])

    def forward(self, x):
        x = self.input_layer(x)
        
        for block in self.res_blocks:
            x = block(x)
            
        # Step 3: Map to final output
        return self.output_layer(x)

class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = SkipMLP(pi_sizes, activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = SkipMLP([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = MLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()
