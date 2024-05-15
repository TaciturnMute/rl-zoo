import torch
import numpy as np
from torch import nn
from finrl_myself.metrics import *
from finrl_myself.srl.ddpg.actor import *
from finrl_myself.srl.ddpg.critic import *
from finrl_myself.noise import get_noise
from finrl_myself.utils import polyak_update
from finrl_myself.buffers.ReplayBuffer import ReplayBuffer
from finrl_myself.logger import episode_logger
from finrl_myself.examine import validate, test
import copy


class DDPG():
    '''
    in training stage, first update SRL model, then update actor and critic.
    '''
    def __init__(
            self,
            env_train=None,
            env_validation=None,
            env_test=None,
            episodes: int = None,
            buffer_size: int = None,
            rl_batch_size: int = None,
            srl_batch_size: int = None,
            tau: float = None,
            gamma: float = None,
            n_steps: int = None,
            if_prioritized: int = None,
            training_start: int = None,
            actor_kwargs: dict = None,
            critic_kwargs: dict = None,
            n_updates: int = None,
            srl_lr: float = None,
            noise_aliase: str = None,
            noise_kwargs: dict = None,
            smooth_noise_aliase: str = None,
            smooth_noise_kwargs: dict = None,
            if_smooth_noise: bool = None,
            print_interval: int = None,
            filename: str = None,
    ):

        self.env_train = env_train
        self.env_validation = env_validation
        self.env_test = env_test
        self.episodes = episodes
        self.rl_batch_size = rl_batch_size
        self.srl_batch_size = srl_batch_size
        self.tau = tau
        self.gamma = gamma
        self.print_interval = print_interval
        self.training_start = training_start
        self.n_updates = n_updates
        self.if_prioritized = if_prioritized
        self.buffer = ReplayBuffer(buffer_capacity=buffer_size,
                                   batch_size=self.rl_batch_size,
                                   n_steps=n_steps,
                                   gamma=gamma,
                                   if_prioritized=if_prioritized)
        self.actor = Actor(**actor_kwargs)
        self.critic = Critic(**critic_kwargs)

        self.srl_model = copy.deepcopy(self.critic.srl_model)
        self.srl_model_optim = get_optimizer(self.critic.optim_aliase, self.srl_model.parameters(), srl_lr)
        self.actor.srl_model.load_state_dict(self.srl_model.state_dict())
        self.critic.srl_model.load_state_dict(self.srl_model.state_dict())

        self.actor_target = Actor(**actor_kwargs)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target = Critic(**critic_kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.if_smooth_noise = if_smooth_noise
        if self.if_smooth_noise:
            smooth_noise_kwargs['batch_size'] = self.batch_size
            self.smooth_noise = get_noise(smooth_noise_aliase, smooth_noise_kwargs)
        self.noise = get_noise(noise_aliase, noise_kwargs)
        self.filename = filename
        self.action_space_range = [self.env_train.action_space.low, self.env_train.action_space.high]

    def replay(self) -> None:
        self.actor_loss_list_once_replay = []
        self.critic_loss_list_once_replay = []
        self.srl_loss_list_once_replay = []
        for _ in range(self.n_updates):
            # I: SRL model update
            data = self.buffer.sample(self.srl_batch_size)
            predictive_transition_prior = self.srl_model.f_pred(data.observations, data.actions)
            assert predictive_transition_prior.shape == data.next_observations.shape
            srl_loss = nn.functional.mse_loss(predictive_transition_prior, data.next_observations)
            self.srl_model_optim.zero_grad()
            srl_loss.backward()
            self.srl_model_optim.step()
            # copy ofe net to actor,critic. not target net. SRL model training is actor's srl and critic's srl training
            self.actor.srl_model.load_state_dict(self.srl_model.state_dict())
            self.critic.srl_model.load_state_dict(self.srl_model.state_dict())

            # II: critic update
            data = self.buffer.sample(self.rl_batch_size)
            with torch.no_grad():
                if self.if_smooth_noise:
                    next_actions = self.actor_target(data.next_observations) + self.smooth_noise()
                    next_actions = next_actions.clamp(torch.tensor(self.action_space_range[0]),
                                                     torch.tensor(self.action_space_range[1]))
                else:
                    next_actions = self.actor_target(data.next_observations)
                next_q_values = self.critic_target(data.next_observations, next_actions) # (batch_size, 1)
                # assert rewards.shape==next_q_values.shape==dones.shape==(self.batch_size,1)
                targets = data.rewards + self.gamma * (1 - data.dones) * next_q_values
                # assert targets.shape==(self.batch_size,1)

            q_value_pres = self.critic(data.observations, data.actions)
            critic_loss = nn.functional.mse_loss(q_value_pres, targets)
            self.critic.optim.zero_grad()
            critic_loss.backward()
            self.critic.optim.step()

            # III actor update
            action_pres = self.actor(data.observations)
            actor_loss = -self.critic(data.observations, action_pres).mean()
            self.actor.optim.zero_grad()
            actor_loss.backward()
            self.actor.optim.step()

            self.actor_loss_list_once_replay.append(actor_loss.detach().numpy())
            self.critic_loss_list_once_replay.append(critic_loss.detach().numpy())
            self.srl_loss_list_once_replay.append(srl_loss.detach().numpy())
            self.logger.total_updates_plus()

    def train(self):
        self.logger = episode_logger()
        for ep in range(1, self.episodes + 1):
            self.logger.reset()
            s = self.env_train.reset()
            done = False
            while not done:
                s_tensor = torch.tensor(s, dtype=torch.float32).reshape((1,) + s.shape)  # add batch dim
                a = self.actor(s_tensor).detach().numpy().reshape(-1)  # (action_dim,)
                a = np.clip(a + self.noise.__call__(), self.action_space_range[0], self.action_space_range[1])
                s_, r, done, _ = self.env_train.step(a)   # ndarray,float,bool,dict
                self.buffer.add(s, a, r, s_, done)
                s = s_
                self.logger.timesteps_plus()

                if self.logger.total_timesteps > self.rl_batch_size and self.logger.total_timesteps > self.training_start:
                    self.replay()
                    # srl model do NOT execute soft updaate
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r,
                                       asset=self.env_train.asset_memory[-1],
                                       time=f'{year[2:]}_{month}_{day}',
                                       actor_loss=np.mean(self.actor_loss_list_once_replay),
                                       critic_loss=np.mean(self.critic_loss_list_once_replay),
                                       srl_loss=np.mean(self.srl_loss_list_once_replay),
                                       )
                else:
                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r,
                                       asset=self.env_train.asset_memory[-1],
                                       time=f'{year[2:]}_{month}_{day}',
                                       actor_loss=0,
                                       critic_loss=0,
                                       srl_loss=0,
                                       )
                if self.logger.timesteps % self.print_interval == 0:
                    self.logger.show(interval=self.print_interval, n_updates=self.n_updates)
            # after one episode:
            validate(self.env_validation, self.actor, self.logger)
            test(self.env_test, self.actor, self.logger)
            self.logger.episode_results_sample()
        # after all episodes:
        self.logger.print_elapsed_time()

    def save(self, filename = None):
        if self.filename is None and filename is None:
            self.filename = 'test'
        elif filename is not None:
            self.filename = filename
        checkpoint = {'actor_state_dict': self.actor.state_dict(),
                      'critic_state_dict': self.critic.state_dict(),
                      # 'actor_optim_state_dict': self.actor.optim.state_dict(),
                      # 'critic_optim_state_dict': self.critic.optim.state_dict()
                      }
        torch.save(checkpoint, self.filename + '.pth')

        self.logger.save(self.filename)

    def load(self, path):
        return self.actor.load_state_dict(torch.load(path)['actor_state_dict'])
