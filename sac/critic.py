import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable
from finrl_myself.weight_initialization import weight_init
from finrl_myself.torch_layers import create_mlp, DualingFore_StateAction
from finrl_myself.optimizers import get_optimizer
from finrl_myself.get_feature_extractors import get_feature_extractor
import collections
from collections import OrderedDict


class Critic(nn.Module):
    def __init__(self,
                 state_dim:int,
                 action_dim:int,
                 activation_fn: Type[nn.Module] = None,
                 feature_extractor_aliase: str = None,
                 feature_extractor_kwargs: Dict = None,
                 net_arch:List = [400,300],
                 optim_aliase:str = None,
                 lr = 3e-4,
                 model_params_init_kwargs = None,
                 n_critics:int = 2,
                 ):
        super(Critic,self).__init__()
        print(f'-----critic------')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net_arch = net_arch   #sequential network architecture, very simple
        self.n_critics = n_critics
        self.activation_fn = activation_fn
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim
        self._setup_model()
        self.optim1 = get_optimizer(optim_aliase, self.qf1.parameters(), lr)
        self.optim2 = get_optimizer(optim_aliase, self.qf2.parameters(), lr)
        weight_init(model_state_dict=self.state_dict(), **model_params_init_kwargs)  # model weights init
        print(f'-----------------')

    def _setup_model(self):
        # critic model
        for idx in range(1, self.n_critics+1):
            qf = nn.Sequential(*create_mlp(input_dim=self.features_dim + self.action_dim,
                                          output_dim=1,
                                          net_arch=self.net_arch,
                                          activation_fn=self.activation_fn))

            self.add_module(f'qf{idx}',qf)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        _inputs = torch.cat((self.feature_extractor(obs), actions), dim=1)
        return (self.qf1(_inputs), self.qf2(_inputs))

    def get_q_values(self, obs: torch.Tensor, actions: torch.Tensor):
        # _inputs = torch.cat((obs, actions),dim=1)
        return self(obs, actions)


class Critic_DualingFore(nn.Module):
    '''
    Dualing MLP, optimize two qf separately
    '''
    def __init__(self,
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
        self.net_arch = net_arch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_critics = n_critics
        self.activation_fn = activation_fn
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim
        self._setup_model()
        self.optim1 = get_optimizer(optim_aliase, self.qf1.parameters(), lr)
        self.optim2 = get_optimizer(optim_aliase, self.qf2.parameters(), lr)
        weight_init(self.parameters(), **model_params_init_kwargs)
        print(f'-----------------')

    def _setup_model(self):
        # two q_network for clipped double q-learning algorithm
        for idx in range(1, self.n_critics+1):
            dualing_mlp = DualingFore_StateAction(
                state_dim=self.features_dim,
                action_dim=self.action_dim,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
            )
            q_net = nn.Linear(dualing_mlp.last_layer_dim_shared_net, 1)
            self.add_module(f'qf{idx}',
                            nn.Sequential(OrderedDict([
                                ('dualing_mlp', dualing_mlp),
                                ('q_net', q_net)]))
                            )

    def get_q_values(self, obs: torch.Tensor, actions: torch.Tensor):
        # _inputs = torch.cat((obs, actions),dim=1)
        return self(obs, actions)

    def forward(self,
                obs: torch.Tensor,
                actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        states = self.feature_extractor(obs)
        latent_feature1 = self.qf1.dualing_mlp(states, actions)
        latent_feature2 = self.qf2.dualing_mlp(states, actions)
        q1 = self.qf1.q_net(latent_feature1)
        q2 = self.qf2.q_net(latent_feature2)
        return (q1, q2)
