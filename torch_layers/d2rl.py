import torch
from torch import nn
from typing import List, Type

class D2RL(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int], activation_fn: Type[nn.Module]):
        super(D2RL,self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activation_fn = activation_fn()
        self.module_list = []

        assert len(hidden_dims) > 0
        self.fc0 = nn.Linear(input_dim, hidden_dims[0])
        self.module_list.append(self.fc0)
        for i in range(0,len(hidden_dims)-1):
            fc = nn.Linear(hidden_dims[i] + input_dim, hidden_dims[i+1])
            self.add_module(f'fc{i+1}', fc)
            self.module_list.append(fc)

    def forward(self, input):
        x = self.module_list[0](input)
        for i in range(0, len(self.hidden_dims)-1):
            x = self.module_list[i+1](torch.concat((x,input), dim=1))
            x = self.activation_fn(x)

        x = torch.concat((x,input), dim=1)
        x = self.activation_fn(x)

        return x

d2rl = D2RL(input_dim=10,hidden_dims=[20,30,40],activation_fn=nn.ReLU)
print(d2rl)
print(d2rl(torch.randn(64,10)).shape)

