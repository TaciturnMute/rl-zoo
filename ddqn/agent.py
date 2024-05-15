# paper: Deep Reinforcement Learning with Double Q-learning
# model: Double DQN
# innovation: Double Q-learning to avoid overestimating

import torch
from torch import nn
from finrl_myself.ddqn.policy import DDQNPolicy
from finrl_myself.buffers.ReplayBuffer import ReplayBuffer
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union,Iterable
from finrl_myself.logger import episode_logger
from finrl_myself.utils import polyak_update


class DDQN():

    def __init__(self,
                 env: None,
                 episodes: int = None,
                 buffer_size: int = None,
                 batch_size: int = None,
                 gamma: float = None,
                 training_start: int = None,
                 policy_kwargs: Dict = None,
                 target_update_interval: int = None,
                 tau: float = None,
                 print_interval: int = 100,
                 filename: str = None,
                ):

        self.env = env
        self.episodes = episodes
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.training_start = training_start
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.print_interval = print_interval
        self.filename = filename
        self.policy = DDQNPolicy(**policy_kwargs)
        self.target_policy = DDQNPolicy(**policy_kwargs)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.buffer = ReplayBuffer(buffer_size,batch_size)
        self.logger = episode_logger()
        self.env_action_range = [-1, 1] # differ from continuous
        self.action_space_range = [0, self.env.action_space.n-1]



    def replay(self) -> None:
        # get one batch samples:
        states, actions, rewards, next_states, dones = self.buffer.sample()
        # I: cumpute td target with target network
        with torch.no_grad():
            next_q_values = self.policy.get_actions_values(next_states)
            _, best_actions = next_q_values.max(dim=1)
            best_actions = best_actions.reshape(actions.shape[0], -1)
            next_q_values_target = self.target_policy.get_actions_values(next_states)
            next_q_values_target = torch.gather(next_q_values_target, dim=1, index=best_actions.long())
            targets = rewards + self.gamma * (1 - dones) * next_q_values_target
        # II: compute action_value estimation
        q_values = self.policy.get_actions_values(states) # (batch_size, action_space_dim)
        q_values = torch.gather(q_values, dim=1, index=actions.long()) # select corresponding value according to action
        # III: gradient discent
        loss = nn.functional.mse_loss(q_values, targets)
        self.policy.optim.zero_grad()
        loss.backward()
        self.policy.optim.step()
        self.logger.record(loss = loss)

    def train(self):
        for ep in range(self.episodes):
            self.logger.reset()
            s = self.env.reset()  # numpy
            done = False
            while not done:
                s_tensor = torch.tensor(s, dtype=torch.float32).reshape(1, -1)
                a = self.policy(s_tensor)  # np.ndarray
                a_env = a * (self.env_action_range[1] - self.env_action_range[0]) / (self.action_space_range[1] - self.action_space_range[0]) + \
                        (-self.action_space_range[0]*self.env_action_range[1] + self.action_space_range[1]*self.env_action_range[0]) / (self.action_space_range[1] - self.action_space_range[0])
                s_, r, done, _ = self.env.step(a_env)
                s = s_

                self.buffer.add(s, a, r, s_, done)
                self.logger.record(record=r, asset=self.env.asset_memory[-1])
                self.logger.timesteps_plus()
                if (self.logger.total_timesteps > self.training_start) and (self.logger.total_timesteps > self.batch_size):
                    self.replay()
                if self.logger.timesteps % self.target_update_interval == 0:
                    polyak_update(self.policy.parameters(), self.target_policy.parameters(), self.tau)
                if self.logger.timesteps % self.print_interval == 0:
                    self.logger.show(interval = self.print_interval)
        self.logger.print_elapsed_time()

    def save(self):
        if self.filename is None:
            self.filename = 'test.pth'
        checkpoint = {'policy_state_dict':self.policy.state_dict(),
                      'policy_optim_state_dict':self.policy.optim.state_dict()
                      }
        torch.save(checkpoint, self.filename)