import torch
from torch import nn
from finrl_myself.weight_initialization import weight_init
from finrl_myself.optimizers import get_optimizer
from finrl_myself.torch_layers import create_mlp,DualingFore_StateAction
from finrl_myself.get_feature_extractors import get_feature_extractor
from typing import Dict, List, Union, Type


class Critic(nn.Module):
    '''
    Sequential MLP
    '''
    def __init__(self,
                 net_arch: List = None,
                 dropout: float = 0,
                 state_dim: int = None,
                 action_dim: int = None,
                 activation_fn: Type[nn.Module] = None,
                 feature_extractor_aliase: str = None,
                 feature_extractor_kwargs: Dict = None,
                 model_params_init_kwargs: Dict = None,
                 optim_aliase: str = None,
                 lr: float = None,

                 ):
        super(Critic, self).__init__()  # 否则会报错：cannot assign module before Module.__init__() call。继承父类属性和方法
        print(f'-----critic------')
        self.net_arch = net_arch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_fn = activation_fn
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim
        self.dropout = dropout
        self._setup_model()
        self.optim = get_optimizer(optim_aliase, self.parameters(), lr)
        weight_init(self.parameters(), **model_params_init_kwargs)
        print(f'-----------------')

    def _setup_model(self):

        self.qf1 = nn.Sequential(
            *create_mlp(
                input_dim=self.features_dim + self.action_dim,
                output_dim=1,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                dropout=self.dropout,
            )

        )

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = self.feature_extractor(obs)
        assert len(states.shape) == len(actions.shape)
        inputs = torch.cat((states, actions), dim=1)
        return self.qf1(inputs)


class Critic_DualingFore(nn.Module):
    '''
    MLP结构，特征提取器为Flatten。

    DualingFore_StateAction
        ***********            ************
         State Net              Action Net
        ***********            ************
             |                      |
         latent_vf              latent_qf
             |______________________|
                        |
              **********************
                 Actor Shared Net
              **********************
                        |
                     q value
    '''
    def __init__(self,
                 net_arch: List[Union[int, Dict[str, List[int]]]] = None,
                 dropout: float = 0,
                 state_dim: int = None,
                 action_dim: int = None,
                 activation_fn: Type[nn.Module] = None,
                 feature_extractor_aliase: str = None,
                 feature_extractor_kwargs: Dict = None,
                 model_params_init_kwargs: Dict = None,
                 optim_aliase: str = None,
                 lr: float = None,
                 ):
        super(Critic_DualingFore, self).__init__()
        print(f'-----critic------')
        self.net_arch = net_arch
        self.dropout = dropout
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_fn = activation_fn
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim
        self._setup_model()
        self.optim = get_optimizer(optim_aliase, self.parameters(), lr)
        weight_init(self.parameters(), **model_params_init_kwargs)
        print(f'-----------------')

    def _setup_model(self):

        self.dualing_mlp = DualingFore_StateAction(
            state_dim=self.features_dim,
            action_dim=self.action_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            dropout=self.dropout,
        )
        self.q_net = nn.Linear(self.dualing_mlp.last_layer_dim_shared_net, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = self.feature_extractor(obs)
        # assert len(states.shape) == len(actions.shape)
        latent_feature = self.dualing_mlp(states,actions)
        return self.q_net(latent_feature)


class Critic_RNNs_Mlp(nn.Module):
    '''
    LSTM encodes states, MLP encodes actions, MLP decodes both of latent features
    '''
    rnn_aliases: dict = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
    }
    def __init__(self,
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
        self.optim = get_optimizer(optim_aliase, self.parameters(), lr)
        weight_init(self.parameters(), **model_params_init_kwargs)
        print(f'-----------------')

    def _setup_model(self):

        self.Rnn_Model = Critic_RNNs_Mlp.rnn_aliases[self.rnn_aliase]
        # lstm encoder
        self.state_net = self.Rnn_Model(
            input_size=self.features_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=False,
        )
        # mlp encoder
        self.action_net = nn.Sequential(
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
        self.qf1 = nn.Sequential(
            *create_mlp(
                input_dim=latent_state_dim + latent_action_dim,
                output_dim=1,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                squash_output=True,
                dropout=self.dropout,
            )
        )

        # When use lstm, the first two dimensions of obs are batch_size and seq_len
        # So flatten will execute from the third dimension.
        # obs dimension: (batch_size, seq_len, (observation_space.shape))
        if self.feature_extractor_aliase == 'flatten':
            self.feature_extractor.flatten.start_dim = 2

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = self.feature_extractor(obs)
        if self.rnn_aliase == 'lstm':
            output, (hn, cn) = self.state_net(states)
        elif self.rnn_aliase == 'gru':
            output, hn = self.state_net(states)
        else:
            raise Exception('rnn_aliase is invalid!')
        latent_state = hn[-1]  # ravel num_layers dimension
        latent_action = self.action_net(actions)

        # assert latent_state.shape[0] == latent_action.shape[0] == obs.shape[0]  # ravel num_layers dimension

        inputs = torch.cat((latent_state, latent_action), dim=1)
        return self.qf1(inputs)
