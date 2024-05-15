import torch
from torch import nn
from finrl_myself.weight_initialization import weight_init
from finrl_myself.optimizers import get_optimizer
from finrl_myself.torch_layers import create_mlp,DualingFore_StateAction
from finrl_myself.get_feature_extractors import get_feature_extractor
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable
from collections import OrderedDict


class Critic(nn.Module):
    '''
    Sequential MLP, optim separately
    '''
    def __init__(self,
                 net_arch: list = [400, 300],
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

        super(Critic, self).__init__()
        print(f'-----critic------')
        self.net_arch = net_arch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_fn = activation_fn
        self.n_critics = n_critics
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
            qf = nn.Sequential(
                *create_mlp(
                    input_dim=self.features_dim + self.action_dim,
                    output_dim=1,
                    net_arch=self.net_arch,
                    activation_fn=self.activation_fn)
            )
            self.add_module(f'qf{idx}',qf)

    def get_qf1_value(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = self.feature_extractor(obs)
        input_ = torch.cat((states, actions), dim=1)
        return self.qf1(input_)

    def forward(self,
                obs: torch.Tensor,
                actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        states = self.feature_extractor(obs)
        input_ = torch.cat((states, actions), dim=1)
        q1 = self.qf1(input_)
        q2 = self.qf2(input_)
        return (q1, q2)


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

    def get_qf1_value(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = self.feature_extractor(obs)
        latent_feature = self.qf1.dualing_mlp(states, actions)
        return self.qf1.q_net(latent_feature)

    def forward(self,
                obs: torch.Tensor,
                actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        states = self.feature_extractor(obs)
        latent_feature1 = self.qf1.dualing_mlp(states, actions)
        latent_feature2 = self.qf2.dualing_mlp(states, actions)
        q1 = self.qf1.q_net(latent_feature1)
        q2 = self.qf2.q_net(latent_feature2)
        return (q1, q2)


class Critic_RNNs_Mlp(nn.Module):
    '''
    LSTM encodes states, MLP encodes actions, MLP decodes both of latent features
    '''
    rnn_aliases: dict = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
    }

    def __init__(
            self,
            n_critic: int = None,
            rnn_aliase: str = None,
            hidden_size: int = None,
            num_layers: int = None,
            dropout: float = None,
            net_arch_action: List[int] = None,
            net_arch: List[int] = None,
            state_dim: int = None,
            action_dim: int = None,
            activation_fn: Type[nn.Module] = None,
            feature_extractor_aliase: str = None,
            feature_extractor_kwargs: Dict = None,
            model_params_init_kwargs: Dict = None,
            optim_aliase: str = None,
            lr: float = None,
    ):
        super(Critic_RNNs_Mlp, self).__init__()  # 否则会报错：cannot assign module before Module.__init__() call。继承父类属性和方法
        print(f'-----critic------')
        self.n_critic = n_critic
        self.rnn_aliase = rnn_aliase
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.net_arch_action = net_arch_action
        self.net_arch = net_arch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_fn = activation_fn
        self.feature_extractor_aliase = feature_extractor_aliase
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim
        self._setup_model()
        self.optim1 = get_optimizer(optim_aliase, self.qf1.parameters(), lr)
        self.optim2 = get_optimizer(optim_aliase, self.qf2.parameters(), lr)
        weight_init(self.parameters(), **model_params_init_kwargs)
        print(f'-----------------')

    def _setup_model(self):
        for idx in range(1, self.n_critic+1):
            self.Rnn_Model = Critic_RNNs_Mlp.rnn_aliases[self.rnn_aliase]

            # lstm encoder
            state_net = self.Rnn_Model(
                input_size=self.features_dim,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bias=True,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=False,
            )

            # mlp encoder
            action_net = nn.Sequential(
                *create_mlp(
                    input_dim=self.action_dim,
                    output_dim=-1,
                    net_arch=self.net_arch_action,
                    activation_fn=self.activation_fn,
                    squash_output=True,
                    dropout=self.dropout,
                )
            )

            latent_state_dim = self.hidden_size
            latent_action_dim = self.net_arch_action[-1] if len(self.net_arch) > 0 else self.action_dim

            # mlp decoder
            q_net = nn.Sequential(
                *create_mlp(
                    input_dim=latent_state_dim + latent_action_dim,
                    output_dim=1,
                    net_arch=self.net_arch,
                    activation_fn=self.activation_fn,
                    squash_output=True,
                    dropout=self.dropout,
                )
            )
            self.add_module(f'qf{idx}',
                            nn.Sequential(OrderedDict([
                                ('state_net', state_net),
                                ('action_net', action_net),
                                ('q_net', q_net)
                            ]))
                            )
            # self.add_module(f'state_net{idx}', state_net)
            # self.add_module(f'action_net{idx}', action_net)
            # self.add_module(f'qf{idx}', q_net)

        # keep seq_len dimension
        # obs dimension: (batch_size, seq_len, (observation_space.shape))
        if self.feature_extractor_aliase == 'flatten':
            self.feature_extractor.flatten.start_dim = 2

    def get_qf1_value(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = self.feature_extractor(obs)
        if self.rnn_aliase == 'lstm':
            output, (hn, cn) = self.qf1.state_net(states)
        elif self.rnn_aliase == 'gru':
            output, hn = self.qf1.state_net(states)
        else:
            raise Exception('rnn_aliase is invalid!')
        latent_state = hn[-1]
        latent_action = self.qf1.action_mean_net(actions)
        assert latent_state.shape[0] == latent_action.shape[0] == obs.shape[0]  # ravel num_layers dimension

        inputs = torch.cat((latent_state, latent_action), dim=1)
        return self.qf1.q_net(inputs)

    def get_qf2_value(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = self.feature_extractor(obs)
        if self.rnn_aliase == 'lstm':
            output, (hn, cn) = self.qf2.state_net(states)
        elif self.rnn_aliase == 'gru':
            output, hn = self.qf2.state_net(states)
        else:
            raise Exception('rnn_aliase is invalid!')
        latent_state = hn[-1]
        latent_action = self.qf2.action_mean_net(actions)
        assert latent_state.shape[0] == latent_action.shape[0] == obs.shape[0]  # ravel num_layers dimension

        inputs = torch.cat((latent_state, latent_action), dim=1)
        return self.qf2.q_net(inputs)

    def forward(
            self,
            obs: torch.Tensor,
            actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1, q2 = self.get_qf1_value(obs, actions), self.get_qf2_value(obs, actions)
        return (q1, q2)

