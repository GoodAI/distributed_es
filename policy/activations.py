import torch
from torch import nn
from torch.distributions import Normal


class GaussianActivation(nn.Module):
    """Gaussian with learneable st_dev, starting at 0.5"""

    st_dev: nn.Parameter

    def __init__(self, size: int):
        super().__init__()
        self.st_dev = nn.Parameter(torch.ones(size, dtype=torch.float32) * 0.5)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        m = Normal(data, torch.exp(self.st_dev))
        out = m.rsample()
        return out


class GaussianFixed(nn.Module):
    """Gaussian with fixed st_dev"""

    st_dev: torch.Tensor

    def __init__(self, size: int):
        super().__init__()
        self.st_dev = torch.ones(size, dtype=torch.float32) * 0.01

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        m = Normal(data, torch.exp(self.st_dev))
        out = m.rsample()
        return out


class DiscreteTanh(nn.Module):

    activation: nn.Module

    def __init__(self):
        super().__init__()
        self.activation = nn.Tanh()

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = self.activation(data)
        data = torch.floor(data * 6) / 6  # 11 bins
        return data


def get_activation(activation: str, size: int) -> nn.Module:
    if activation == 'none' or activation == 'None':
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'discrete':
        return DiscreteTanh()
    elif activation == 'relu':
        return nn.LeakyReLU()
    elif activation == 'gaussian':
        return GaussianActivation(size)
    elif activation == 'gaussian_fixed':
        return GaussianFixed(size)
    raise Exception('unsupported activation')
