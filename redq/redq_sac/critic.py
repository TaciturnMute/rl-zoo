import torch
from torch import nn
from typing import Dict, List, Tuple, Type, Union
from finrl_myself.weight_initialization import weight_init
from finrl_myself.torch_layers import create_mlp,DualingFore_StateAction
from finrl_myself.optimizers import get_optimizer
from finrl_myself.get_feature_extractors import get_feature_extractor
import collections
from collections import OrderedDict


class Critic(nn.Module):

    def __init__(
            self,
            N: int,
            state_dim: int,
            action_dim: int,
            activation_fn: Type[nn.Module] = None,
            feature_extractor_aliase: str = None,
            feature_extractor_kwargs: dict = None,
            net_arch: list = [400,300],
            optim_aliase: str = None,
            lr: float = 3e-4,
            model_params_init_kwargs: dict = None,
    ):
        super(Critic, self).__init__()
        print(f'-----critic------')
        self.N = N
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_fn = activation_fn
        self.net_arch = net_arch    # sequential network architecture, very simple
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim
        self._setup_model()
        self.optim_list = []
        for index in range(N):
            self.optim_list.append(get_optimizer(optim_aliase, self.qf_list[index].parameters(), lr))
        weight_init(model_state_dict=self.state_dict(), **model_params_init_kwargs)  # model weights init
        print(f'-----------------')

    def _setup_model(self):
        # critic model
        self.qf_list = []
        for idx in range(1, self.N + 1):
            q = nn.Sequential(
                *create_mlp(
                    input_dim=self.features_dim + self.action_dim,
                    output_dim=1,
                    net_arch=self.net_arch,
                    activation_fn=self.activation_fn
                )
            )
            self.qf_list.append(q)
            self.add_module(f'q{idx}', q)

    def get_M_values(self, indexs: list, obs: torch.Tensor, actions: torch.Tensor) -> List[torch.Tensor]:
        '''

        :param indexs: M Q functions' indexs
        :param obs:
        :param actions:
        :return: [tensor(), tensor(),..., tensor()] M length
                 tensor() shape is (batch_size, 1)
        '''
        _inputs = torch.cat((self.feature_extractor(obs), actions), dim=1)
        q_values_list = [self.qf_list[i](_inputs) for i in indexs]
        return q_values_list

    def forward(self, obs, actions) -> List[torch.Tensor]:
        _inputs = torch.cat((self.feature_extractor(obs), actions), dim=1)
        q_values_list = [self.qf_list[i](_inputs) for i in range(self.N)]
        return q_values_list


class Critic_DualingFore(nn.Module):
    '''
    Dualing MLP, optimize two qf separately
    '''
    def __init__(self,
                 N: int = None,
                 net_arch: List[Union[int, Dict[str, List[int]]]] = None,
                 state_dim: int = None,
                 action_dim: int = None,
                 activation_fn: Type[nn.Module] = None,
                 feature_extractor_aliase: str = None,
                 feature_extractor_kwargs: Dict = None,
                 model_params_init_kwargs: dict = None,
                 optim_aliase: str = None,
                 lr: float = None,
                 n_critics: int = 2
                 ):

        super(Critic_DualingFore, self).__init__()
        print(f'-----critic------')
        self.N = N
        self.net_arch = net_arch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_critics = n_critics
        self.activation_fn = activation_fn
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim
        self._setup_model()
        self.optim_list = []
        for index in range(N):
            # optimize q function separately
            self.optim_list.append(get_optimizer(optim_aliase, self.qf_list[index].parameters(), lr))
        weight_init(self.parameters(), **model_params_init_kwargs)
        print(f'-----------------')

    def _setup_model(self):

        self.qf_list = []
        for idx in range(1, self.N + 1):
            dualing_mlp = DualingFore_StateAction(
                state_dim=self.features_dim,
                action_dim=self.action_dim,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
            )
            q_net = nn.Linear(dualing_mlp.last_layer_dim_shared_net, 1)
            qf = nn.Sequential(OrderedDict([
                                ('dualing_mlp', dualing_mlp),
                                ('q_net', q_net)]))
            self.add_module(f'qf{idx}',qf)
            self.qf_list.append(qf)

    def get_M_values(self, indexs: list, obs: torch.Tensor, actions: torch.Tensor) -> List[torch.Tensor]:
        '''

        :param indexs: M Q functions' indexs
        :param obs:
        :param actions:
        :return: [tensor(), tensor(),..., tensor()] M length
                 tensor() shape is (batch_size, 1)
        '''
        states = self.feature_extractor(obs)
        qf_list_temp = [self.qf_list[i] for i in indexs]
        latent_features_list = [qf_list_temp[i].dualing_mlp(states, actions) for i in range(len(indexs))]
        q_values_list = [qf_list_temp[i].q_net(latent_features_list[i]) for i in range(len(indexs))]
        return q_values_list

    def forward(self, obs, actions) -> List[torch.Tensor]:
        states = self.feature_extractor(obs)
        latent_features_list = [self.qf_list[i].dualing_mlp(states, actions) for i in range(self.N)]
        q_values_list = [self.qf_list[i].q_net(latent_features_list[i]) for i in range(self.N)]
        return q_values_list
