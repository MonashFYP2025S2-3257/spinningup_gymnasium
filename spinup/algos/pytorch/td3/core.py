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
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


class ResidualBlock(nn.Module):
    """
    A residual hidden layer.

    It calculates:
        output = activation(Linear(x) + x)

    This requires the input and output dimensions to be the same.
    """
    def __init__(self, size, activation):
        super().__init__()

        self.linear = nn.Linear(size, size)
        self.act = activation()

    def forward(self, x):
        return self.act(self.linear(x) + x)


class SkipMLP(nn.Module):
    """
    MLP with residual/skip hidden layers.

    Structure:
        Input layer:
            Linear(input_dim -> hidden_dim) + activation

        Residual hidden layers:
            activation(Linear(x) + x)

        Output layer:
            Linear(hidden_dim -> output_dim) + output_activation
    """
    def __init__(self, sizes, activation, output_activation=nn.Identity):
        super().__init__()

        assert len(sizes) >= 3, "sizes must be [input_dim, hidden_dim, output_dim] or deeper"

        input_dim = sizes[0]
        hidden_dim = sizes[1]
        output_dim = sizes[-1]

        # First layer maps from input dimension to hidden dimension
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation()
        )

        # Residual blocks operate only in hidden_dim space
        # Number of residual blocks should match the number of hidden layers after the first one
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, activation)
            for _ in range(len(sizes) - 3)
        ])

        # Final layer maps from hidden dimension to output dimension
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            output_activation()
        )

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.res_blocks:
            x = block(x)

        x = self.output_layer(x)

        return x


class MLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()

        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]

        self.pi = SkipMLP(
            sizes=pi_sizes,
            activation=activation,
            output_activation=nn.Tanh
        )

        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits
        return self.act_limit * self.pi(obs)


class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()

        q_sizes = [obs_dim + act_dim] + list(hidden_sizes) + [1]

        self.q = SkipMLP(
            sizes=q_sizes,
            activation=activation,
            output_activation=nn.Identity
        )

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))

        # Critical to ensure q has right shape
        return torch.squeeze(q, -1)


class MLPActorCritic(nn.Module):

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes=(256, 256),
        activation=nn.ReLU
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # Build policy and Q-functions
        self.pi = MLPActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            act_limit=act_limit
        )

        self.q1 = MLPQFunction(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation
        )

        self.q2 = MLPQFunction(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            activation=activation
        )

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()