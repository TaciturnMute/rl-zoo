# paper: Addressing Function Approximation Error in Actor-Critic Methods
# model: Twin Delayed Deep Deterministic policy gradient algorithm (TD3)
# innovation: clipped double Q-learning; delayed policy update; target policy smooth regularization;

import torch
import numpy as np
from torch import nn
from finrl_myself.noise import get_noise
from finrl_myself.utils import polyak_update
from finrl_myself.buffers.ReplayBuffer import ReplayBuffer
from finrl_myself.td3.actor import *
from finrl_myself.td3.critic import *
from finrl_myself.logger import episode_logger
from finrl_myself.metrics import *
from finrl_myself.examine import validate, test
import copy
from copy import deepcopy

class TD3():

    def __init__(self,
                 env_train = None,
                 env_validation = None,
                 env_test = None,
                 buffer_size: int = None,
                 batch_size:int = None,
                 episodes: int = None,
                 n_updates: int = None,
                 n_steps: int = None,
                 if_prioritized: int = None,
                 tau: float = None,
                 gamma: float = None,
                 policy_update_delay: int = None,  # 2
                 target_copy_interval: int = None,
                 training_start: int = None,
                 actor_aliase: str = None,
                 critic_aliase: str = None,
                 critic_kwargs: dict = None,
                 actor_kwargs: dict = None,
                 noise_aliase: str = None,
                 noise_kwargs: dict = None,
                 smooth_noise_aliase: str = None,
                 smooth_noise_kwargs: dict = None,
                 print_interval: int = None,
                 filename: str = None,
                 ):
        # from here get specific actor and critic
        actor_aliases: dict = {
            "SequentialMlp": Actor,
            "RNNs": Actor_RNNs,
        }
        critic_aliases: dict = {
            "SequentialMlp": Critic,
            "DualingFore": Critic_DualingFore,
            "RNNs_Mlp": Critic_RNNs_Mlp,
        }
        self.env_train = env_train
        self.env_validation = env_validation
        self.env_test = env_test
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_capacity=buffer_size,
                                   batch_size=batch_size,
                                   n_steps=n_steps,
                                   gamma=gamma,
                                   if_prioritized=if_prioritized)
        self.episodes = episodes
        self.n_updates = n_updates
        self.n_steps = n_steps
        self.if_prioritized = if_prioritized
        self.tau = tau
        self.policy_update_delay = policy_update_delay
        self.target_copy_interval = target_copy_interval
        self.gamma = gamma
        self.training_start = training_start
        self.print_interval = print_interval
        self.filename = filename

        self.Actor = actor_aliases[actor_aliase]
        self.actor = self.Actor(**actor_kwargs)
        self.actor_target = self.Actor(**actor_kwargs)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.Critic = critic_aliases[critic_aliase]
        self.critic = self.Critic(**critic_kwargs)
        self.critic_target = self.Critic(**critic_kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.noise = get_noise(noise_aliase, noise_kwargs)
        smooth_noise_kwargs['batch_size'] = self.batch_size
        self.smooth_noise = get_noise(smooth_noise_aliase, smooth_noise_kwargs)
        self.action_space_range = [self.env_train.action_space.low, self.env_train.action_space.high]

    def replay(self) -> None:
        self.actor_loss_list_once_replay = []
        self.critic_loss_list_once_replay = []
        for _ in range(self.n_updates):
            data = self.buffer.sample()
            # I update critic:
            with torch.no_grad():
                next_actions = self.actor_target(data.next_observations) + self.smooth_noise.__call__()
                next_actions = next_actions.clamp(torch.tensor(self.action_space_range[0]), torch.tensor(self.action_space_range[1]))
                next_q1, next_q2 = self.critic_target(data.next_observations, next_actions)
                # assert rewards.shape==dones.shape==torch.min(next_q1,next_q2).shape==(self.batch_size, 1)
                td_targets = data.rewards + self.gamma * (1 - data.dones) * torch.min(next_q1, next_q2)
            q_value_pres1, q_value_pres2 = self.critic(data.observations, data.actions)  # tensor

            if self.if_prioritized:
                # 1.cal td errors
                td_errors = (q_value_pres1 + q_value_pres2) / self.critic.n_critics - td_targets
                td_errors = td_errors.detach().numpy()
                # td_errors = q_value_pres1 - td_targets
                alpha = self.buffer.alpha
                # 2. update priorities
                self.buffer.update_priorities([*zip(data.sample_idx, (abs(td_errors)**alpha))])
                weights = (data.sample_probs / min(data.sample_probs))**(-self.buffer.beta())
                assert weights.requires_grad == False
                critic_loss1 = (((q_value_pres1 - td_targets) ** 2) * (weights / 2)).mean()
                critic_loss2 = (((q_value_pres2 - td_targets) ** 2) * (weights / 2)).mean()
                critic_loss = critic_loss1 + critic_loss2
            else:
                critic_loss = nn.functional.mse_loss(q_value_pres1, td_targets) + nn.functional.mse_loss(q_value_pres2,
                                                                                                         td_targets)
            # optim respectively
            self.critic.optim1.zero_grad()
            self.critic.optim2.zero_grad()
            critic_loss.backward()
            self.critic.optim1.step()
            self.critic.optim2.step()

            # II update actor:
            if self.logger.total_updates % self.policy_update_delay == 0:
                action_pres = self.actor(data.observations)  # tensor, cumpute by learning net actor
                actor_loss = -self.critic.get_qf1_value(data.observations, action_pres).mean()  # take derivative of the actor_loss, get ddpg policy gradient
                self.actor.optim.zero_grad()
                actor_loss.backward()
                self.actor.optim.step()
                self.actor_loss_list_once_replay.append(actor_loss.detach().numpy())
            self.critic_loss_list_once_replay.append(critic_loss.detach().numpy())
            # note: in td3 paper, target_copy_delay is equal to policy_update_delay
            # it means no matter actor training frequency or actor/critic soft update, the frequency is lag behind
            if self.logger.total_updates % self.target_copy_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
            self.logger.total_updates_plus()

    def train(self):
        self.logger = episode_logger()
        for ep in range(1, self.episodes+1):
            s = self.env_train.reset()
            done = False
            self.logger.reset()
            while not done:
                s_tensor = torch.tensor(s, dtype=torch.float32).reshape((1,) + s.shape)
                a = self.actor(s_tensor).detach().numpy().reshape(-1)
                a = np.clip(a + self.noise.__call__(), self.action_space_range[0], self.action_space_range[1])
                s_, r, done, _ = self.env_train.step(a)  # ndarray,float,bool,empty_dict
                self.buffer.add(s, a, r, s_, done)  # ndarray,ndarray,scale,list,bool
                s = s_
                self.logger.timesteps_plus()

                if self.logger.total_timesteps > self.batch_size and self.logger.total_timesteps > self.training_start:
                    self.replay()
                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r,
                                       asset=self.env_train.asset_memory[-1],
                                       time=f'{year[2:]}_{month}_{day}',
                                       critic_loss=np.mean(self.critic_loss_list_once_replay),
                                       )
                    if len(self.actor_loss_list_once_replay) > 0:
                        # in case in one replay, actor can not be updated
                        self.logger.record(actor_loss=np.mean(self.actor_loss_list_once_replay))
                else:
                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r,
                                       asset=self.env_train.asset_memory[-1],
                                       time=f'{year[2:]}_{month}_{day}',
                                       actor_loss=0,
                                       critic_loss=0,
                                       )
                if self.logger.timesteps % self.print_interval == 0:
                    self.logger.show(policy_update_delay=self.policy_update_delay,
                                     interval=self.print_interval,
                                     n_updates=self.n_updates)
            validate(self.env_validation, self.actor, self.logger)
            test(self.env_test, self.actor, self.logger)
            self.logger.episode_results_sample()
        self.logger.print_elapsed_time()

    def save(self, filename = None):
        if self.filename is None and filename is None:
            self.filename = 'test'
        elif filename is not None:
            self.filename = filename
        checkpoint = {'actor_state_dict': self.actor.state_dict(),
                      'critic_state_dict': self.critic.state_dict(),
                      'actor_optim_state_dict': self.actor.optim.state_dict(),
                      'critic_optim_state_dict1': self.critic.optim1.state_dict(),
                      'critic_optim_state_dict2': self.critic.optim2.state_dict()
        }
        torch.save(checkpoint, self.filename + '.pth')

        self.logger.save(self.filename)

    def load(self, path):
        return self.actor.load_state_dict(torch.load(path)['actor_state_dict'])
