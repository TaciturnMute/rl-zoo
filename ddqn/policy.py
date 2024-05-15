import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable
from finrl_myself.torch_layers import create_mlp
from finrl_myself.optimizers import get_optimizer
from finrl_myself.weight_initialization import weight_init
from finrl_myself.schedules import get_schedule
import numpy as np

class DDQNPolicy(nn.Module):

    def __init__(self,
                 net_arch: List = [64, 64],
                 state_dim: int = None,
                 action_dim: int = None,
                 optim_aliase: str = None,
                 lr: float = None,
                 model_params_init_kwargs = None,
                 epsilon_schedule_aliase: str = None,
                 epsilon_schedule_kwargs: Dict = None,
                 ):

        super(DDQNPolicy, self).__init__()
        self.net_arch = net_arch
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_params_init_kwargs = model_params_init_kwargs
        self.epsilon_schedule_aliase = epsilon_schedule_aliase
        self._setup_model()
        print(f'------ddqn policy------')
        self.optim = get_optimizer(optim_aliase, self.parameters(), lr)
        weight_init(model_state_dict=self.state_dict(), **self.model_params_init_kwargs)  # model weights init
        self.epsilon_schedule = get_schedule(self.epsilon_schedule_aliase, epsilon_schedule_kwargs)
        print(f'----------------------')


    def _setup_model(self) -> None:
        self.q_net = nn.Sequential(*create_mlp(self.state_dim, self.action_dim, self.net_arch))

    def get_actions_values(self,
                           states: torch.Tensor) -> torch.Tensor:
        # return (batch_size, action_space_dim)
        return self.q_net(states)

    def get_actions(self, states: torch.Tensor) -> torch.Tensor:
        # This function generate action with greedy_algorithm
        # Forward function generate action with epsilon-greedy algorithm
        q_values = self.get_actions_values(states).reshape(1, -1)
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def forward(self, states: torch.Tensor) -> np.ndarray:
        q_values = self.get_actions_values(states).reshape(1, -1)  # two dimension
        if self.epsilon_schedule is not None:
            epsilon = self.epsilon_schedule.get_value()
            action = int(q_values.argmax(dim = 1).reshape(-1).detach().numpy()) if np.random.rand() > epsilon else np.random.randint(0, q_values.shape[1] - 1)
        else:
            action = int(q_values.argmax(dim=1).reshape(-1).detach().numpy())
        return np.array([action])