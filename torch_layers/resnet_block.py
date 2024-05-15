import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable


class ResnetBlock1(nn.Module):
    '''
    output = activate(input + linear(input))
    '''
    def __init__(self, input_dim: int, output_dim: int, activation_fn: Type[nn.Module]):
        super(ResnetBlock1,self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.af = activation_fn()

    def forward(self, input) -> torch.Tensor:
        x = self.fc(input)
        input_dim = int(input.shape[1])
        feature_dim = int(x.shape[1])
        # pad
        if input_dim > feature_dim:
            pad_dim = input_dim - feature_dim
            x = F.pad(input=x, pad=[0, pad_dim], mode='constant', value=1)
        elif input_dim < feature_dim:
            pad_dim = feature_dim - input_dim
            input = F.pad(input=input, pad=[0, pad_dim], mode='constant', value=1)
        # assert input.shape == x.shape
        x += input
        x = self.af(x)
        return x


class ResnetBlock2(nn.Module):
    '''
    output = activate(input + linear(activate(linear(input))))
    '''
    def __init__(self, input_dim: int, hidden_dims: List[int], activation_fn: Type[nn.Module]):
        super(ResnetBlock2,self).__init__()
        self.fc0 = nn.Linear(input_dim, hidden_dims[0])
        self.fc1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.af = activation_fn()

    def forward(self, input):
        x = self.fc0(input)
        x = self.af(x)
        x = self.fc1(x)
        input_dim = int(input.shape[1])
        feature_dim = int(x.shape[1])
        if input_dim > feature_dim:
            pad_dim = input_dim - feature_dim
            x = F.pad(input=x, pad=[0, pad_dim], mode='constant', value=1)
        elif input_dim < feature_dim:
            pad_dim = feature_dim - input_dim
            input = F.pad(input=input, pad=[0, pad_dim], mode='constant', value=1)
        x += input
        x = self.af(x)
        return x

# resnetblock = ResnetBlock1(input_dim=10, output_dim=20, activation_fn=nn.ReLU)
# print(resnetblock)
# print(resnetblock(torch.randn(64,10)))

# resnetblock = ResnetBlock2(input_dim=10, hidden_dims=[20, 30], activation_fn=nn.Tanh)
# print(resnetblock)
# print(resnetblock(torch.randn(64,10)))

