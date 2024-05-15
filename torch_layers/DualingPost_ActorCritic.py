import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable


class DualingPost_ActorCritic(nn.Module):
    '''
    policy 表示生成latent_pi的一支
    value 表示生成latent_vf的一支
    '''
    def __init__(self,
                 net_arch: List[Union[int, Dict[str, List[int]]]] = None,
                 input_dim: int = None,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 ):
        super(DualingPost_ActorCritic, self).__init__()

        # Iterate through the shared layers and build the shared parts of the network
        shared_net = []
        last_layer_dim_shared_net = input_dim
        policy_only_layers = []
        value_only_layers = []
        for layer in net_arch:
            if isinstance(layer, int):  # Check that this is a shared layer
                # TODO: give layer a meaningful name
                shared_net.append(nn.Linear(last_layer_dim_shared_net, layer))  # add linear of size layer
                shared_net.append(activation_fn())
                last_layer_dim_shared_net = layer
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                    policy_only_layers = layer["pi"]

                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                    value_only_layers = layer["vf"]
                break  # From here on the network splits up in policy and value network

        # get value_net from value_only_layers, get policy_net from policy_only layers
        value_net = []
        policy_net = []

        last_layer_dim_value_net = last_layer_dim_shared_net
        last_layer_dim_policy_net = last_layer_dim_shared_net
        # value_only_layers maybe null list
        if len(value_only_layers) > 0:
            last_layer_dim_value_net = last_layer_dim_shared_net  # value net input dim
            for vf_layer_dim in value_only_layers:
                assert isinstance(vf_layer_dim, int), "Error: net_arch[-1]['vf'] must only contain integers."
                value_net.append(nn.Linear(last_layer_dim_value_net,vf_layer_dim)) # add Linear layer
                value_net.append(activation_fn())  #add activation function
                last_layer_dim_value_net = vf_layer_dim

        if len(policy_only_layers) > 0:
            last_layer_dim_policy_net = last_layer_dim_shared_net  # policy net input dim
            for pi_layer_dim in policy_only_layers:
                assert isinstance(pi_layer_dim, int), "Error: net_arch[-1]['pi'] must only contain integers."
                policy_net.append(nn.Linear(last_layer_dim_policy_net,pi_layer_dim))
                policy_net.append(activation_fn())
                last_layer_dim_policy_net = pi_layer_dim

        self.latent_vf_dim = last_layer_dim_value_net
        self.latent_pi_dim = last_layer_dim_policy_net

        self.shared_net = nn.Sequential(*shared_net)
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)

    def forward(self,states:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        # policy forward use this
        shared_latent = self.shared_net(states)
        latent_vf = self.value_net(shared_latent)
        latent_pi = self.policy_net(shared_latent)
        return (latent_pi,latent_vf)

    def forward_critic(self,states:torch.Tensor) -> torch.Tensor:
        # only get latent representation for value network
        return self.value_net(self.shared_net(states))

    def forward_actor(self,states:torch.Tensor) -> torch.Tensor:
        # only get latent representation for policy network
        return self.policy_net(self.shared_net(states))