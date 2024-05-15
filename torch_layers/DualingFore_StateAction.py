import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable


class DualingFore_StateAction(nn.Module):
    '''
    vf 表示处理状态的一支
    qf 表示处理动作的一支
    '''
    def __init__(self,
                 state_dim,
                 action_dim,
                 net_arch: List[Union[int, Dict[str,List[int]]]] = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 dropout: float = 0,
                 ):
        super(DualingFore_StateAction, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_arch = net_arch
        self.dropout = dropout

        # in case of the net arch is None
        shared_net = []
        vf_only_layers = []
        qf_only_layers = []
        last_layer_dim_vf_net = state_dim
        last_layer_dim_qf_net = action_dim
        self.last_layer_dim_shared_net = state_dim + action_dim  # in case state_only_layers and action_only_layers is None

        for layer in net_arch:
            # From here on the network splits up in policy and value network
            if isinstance(layer, dict):
                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[0]['vf'] must contain a list of integers."
                    vf_only_layers = layer["vf"]
                    if len(layer['vf']) > 0:
                        last_layer_dim_vf_net = layer["vf"][-1]

                if "qf" in layer:
                    assert isinstance(layer["qf"], list), "Error: net_arch[1]['qf'] must contain a list of integers."
                    qf_only_layers = layer["qf"]
                    if len(layer['qf']) > 0:
                        last_layer_dim_qf_net = layer["qf"][-1]
                self.last_layer_dim_shared_net = last_layer_dim_vf_net + last_layer_dim_qf_net
            else:
                assert isinstance(layer, int), "Error: the net_arch list can only contain ints and dicts"
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(self.last_layer_dim_shared_net, layer))  # add linear of size layer
                shared_net.append(activation_fn())
                if self.dropout > 0:
                    shared_net.append(nn.Dropout(self.dropout))
                self.last_layer_dim_shared_net = layer

        # get vf_net from vf_only_layers
        vf_net = []
        if len(vf_only_layers) > 0:
            last_layer_dim = state_dim
            for layer in vf_only_layers:
                assert isinstance(layer, int), "Error: net_arch[0]['vf'] must only contain integers."
                vf_net.append(nn.Linear(last_layer_dim, layer))  # add Linear layer
                vf_net.append(activation_fn())
                last_layer_dim = layer

        # get qf_net from qf_only_layers
        qf_net = []
        if len(qf_only_layers) > 0:
            last_layer_dim = action_dim
            for layer in qf_only_layers:
                assert isinstance(layer, int), "Error: net_arch[0]['qf'] must only contain integers."
                qf_net.append(nn.Linear(last_layer_dim, layer))  # add Linear layer
                qf_net.append(activation_fn())
                last_layer_dim = layer

        self.latent_dim = self.last_layer_dim_shared_net
        self.vf_net = nn.Sequential(*vf_net)
        self.qf_net = nn.Sequential(*qf_net)
        self.shared_net = nn.Sequential(*shared_net)

    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor
                ):

        latent_vf = self.vf_net(states)
        latent_qf = self.qf_net(actions)
        shared_latent = torch.cat((latent_vf, latent_qf), dim=1)
        return self.shared_net(shared_latent)