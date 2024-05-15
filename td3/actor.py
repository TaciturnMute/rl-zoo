import torch
from torch import nn
from finrl_myself.weight_initialization import weight_init
from finrl_myself.optimizers import get_optimizer
from finrl_myself.torch_layers import create_mlp
from finrl_myself.get_feature_extractors import get_feature_extractor
from typing import Dict, List, Type

class Actor(nn.Module):

    def __init__(self,
                 net_arch: List = None,
                 state_dim: int = None,
                 action_dim: int = None,
                 activation_fn: Type[nn.Module] = None,
                 feature_extractor_aliase: str = None,
                 feature_extractor_kwargs: Dict = None,
                 model_params_init_kwargs: dict = None,
                 optim_aliase: str = None,
                 lr: float = None,

                 ):
        super(Actor, self).__init__()
        print(f'------actor------')
        self.net_arch = net_arch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_fn = activation_fn
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim  # FlattenExtractor will flatten and return flatten dim
        self._setup_model()
        self.optim = get_optimizer(optim_aliase, self.parameters(), lr)
        weight_init(model_state_dict = self.state_dict(), **model_params_init_kwargs)
        print(f'-----------------')

    def _setup_model(self) -> None:
        self.mu = nn.Sequential(*create_mlp(input_dim=self.features_dim,
                                            output_dim=self.action_dim,
                                            net_arch=self.net_arch,
                                            activation_fn=self.activation_fn,
                                            squash_output=True)
                                )

    def get_actions(self, obs: torch.Tensor, deterministic=True) -> torch.Tensor:
        return self(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mu(self.feature_extractor(obs))


class Actor_RNNs(nn.Module):
    '''
    RNNs actor
    '''
    rnn_aliases: dict = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
    }
    def __init__(self,
                 rnn_aliase: str = None,
                 hidden_size: int = None,
                 num_layers: int = None,
                 dropout: float = 0,
                 net_arch: List[int] = None,
                 state_dim: int = None,
                 action_dim: int = None,
                 activation_fn: Type[nn.Module] = None,
                 feature_extractor_aliase: str = None,
                 feature_extractor_kwargs: dict = None,
                 model_params_init_kwargs: dict = None,
                 optim_aliase: str = None,
                 lr: float = None,
                 ):
        super(Actor_RNNs, self).__init__()
        print(f'------actor------')
        self.rnn_aliase = rnn_aliase
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.net_arch = net_arch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.activation_fn = activation_fn
        self.feature_extractor_aliase = feature_extractor_aliase
        self.feature_extractor = get_feature_extractor(feature_extractor_aliase)
        self.feature_extractor = self.feature_extractor(**feature_extractor_kwargs)
        self.features_dim = self.feature_extractor.features_dim  # FlattenExtractor will flatten and return flatten dim
        self._setup_model()
        self.optim = get_optimizer(optim_aliase, self.parameters(), lr)
        weight_init(model_state_dict=self.state_dict(), **model_params_init_kwargs)
        print(f'-----------------')

    def _setup_model(self):
        self.Rnn_Model = Actor_RNNs.rnn_aliases[self.rnn_aliase]
        self.rnn_net = self.Rnn_Model(
            input_size=self.features_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=False,
        )
        self.mu = nn.Sequential(
            *create_mlp(
                input_dim=self.hidden_size,
                output_dim=self.action_dim,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                squash_output=True,
                dropout=self.dropout,
            )
        )
        # obs dimension: (batch_size, seq_len, (observation_space.shape))
        if self.feature_extractor_aliase == 'flatten':
            self.feature_extractor.flatten.start_dim = 2

    def get_actions(self, obs: torch.Tensor, deterministic=True):
        return self(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        rnn_input = self.feature_extractor(obs)
        if self.rnn_aliase == 'lstm':
            output, (hn, cn) = self.rnn_net(rnn_input)
        elif self.rnn_aliase == 'gru':
            output, hn = self.rnn_net(rnn_input)
        else:
            raise Exception('rnn_aliase is invalid!')
        return self.mu(hn[-1])  # ravel num_layers dimension