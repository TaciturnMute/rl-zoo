import torch
from torch import nn
from typing import List


def create_mlp(input_dim, hidden_dims, output_dim, activation_fn, dropout, squash_output) -> List[nn.Module]:

    def mlp_block(input_dim, output_dim, activation_fn, dropout) -> List[nn.Module]:
        return [nn.Linear(input_dim,output_dim),activation_fn(), nn.Dropout(dropout)]

    module = []
    last_dim = input_dim
    if len(hidden_dims) > 0:
        module.extend(mlp_block(input_dim, hidden_dims[0], activation_fn, dropout))
        last_dim = hidden_dims[0]
        for i in range(1, len(hidden_dims)):
            module.extend(mlp_block(last_dim, hidden_dims[i], activation_fn, dropout))
            last_dim = hidden_dims[i]

    if output_dim != -1:
        module.append(nn.Linear(last_dim, output_dim, activation_fn))

    if squash_output:
        module.append(nn.Tanh())

    return module

# print(nn.Sequential(*create_mlp(input_dim=11,
#                                 hidden_dims=[64,64],
#                                 output_dim=1,
#                                 activation_fn=nn.ReLU,
#                                 dropout=0.2,
#                                 squash_output=True)))