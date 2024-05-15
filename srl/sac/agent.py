import torch
from torch import nn
import numpy as np
import copy
from finrl_myself.srl.sac.actor import *
from finrl_myself.srl.sac.critic import *
from finrl_myself.buffers.ReplayBuffer import ReplayBuffer
from finrl_myself.utils import polyak_update
from finrl_myself.logger import episode_logger
from finrl_myself.examine import validate, test
import time

# paper: Soft Actor-Critic Algorithms and Applications
# model: Soft Actor-Critic
# innovation: soft value function, maximum entropy RL, auto-adaptive entropy coefficient,

class SAC():
    def __init__(
            self,
            env_train,
            env_validation,
            env_test,
            episodes: int = 10,
            n_updates: int = None,
            n_steps: int = None,
            if_prioritized: int = None,
            buffer_size: int = 100000,
            rl_batch_size: int = 100,
            srl_batch_size: int = 100,
            actor_aliase: str = None,
            critic_aliase: str = None,
            train_start: int = 100,
            target_update_interval: int = 1,
            policy_update_delay: int = 1,
            gamma: float = 0.99,
            tau: float = 0.005,
            srl_lr: float = None,
            filename: str = None,
            init_value: float = 1.0,
            ent_coef_lr: float = 3e-4,
            auto_ent_coef: bool = True,
            critic_kwargs: dict = None,
            actor_kwargs: dict = None,
            print_interval: int = 100,
    ):

        self.env_train = env_train
        self.env_validation = env_validation
        self.env_test = env_test
        self.gamma = gamma
        self.rl_batch_size = rl_batch_size
        self.srl_batch_size = srl_batch_size
        self.episodes = episodes
        self.n_updates = n_updates
        self.n_steps = n_steps
        self.if_prioritized = if_prioritized
        self.tau = tau
        self.train_start = train_start
        self.policy_update_delay = policy_update_delay  # actor update delay
        self.target_update_interval = target_update_interval  # target_net update delay
        self.init_value = init_value
        self.auto_ent_coef = auto_ent_coef
        self.filename = filename  # save path
        self.print_interval = print_interval
        self.critic = Critic(**critic_kwargs)
        self.actor = Actor(**actor_kwargs)
        self.srl_model = copy.deepcopy(self.critic.srl_model)
        self.srl_model_optim = get_optimizer(self.critic.optim_aliase, self.srl_model.parameters(), srl_lr)
        self.actor.srl_model.load_state_dict(self.srl_model.state_dict())
        self.critic.srl_model.load_state_dict(self.srl_model.state_dict())
        self.actor_target = Actor(**actor_kwargs)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target = Critic(**critic_kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.buffer = ReplayBuffer(buffer_capacity=buffer_size,
                                   batch_size=self.rl_batch_size,
                                   n_steps=n_steps,
                                   gamma=gamma,
                                   if_prioritized=if_prioritized)
        self.action_range = [self.env_train.action_space.low,self.env_train.action_space.high]
        self.n_updates_now = 0

        if self.auto_ent_coef:
            self.target_entropy = -np.prod(self.env_train.action_space.shape).astype(np.float32)  # float
            self.log_ent_coef = torch.log(torch.ones(1) * self.init_value).requires_grad_(True) # torch.float
            self.ent_coef_optim = torch.optim.Adam([self.log_ent_coef], ent_coef_lr)
        else:
            self.log_ent_coef = torch.log(torch.ones(1) * init_value)  # torch.float
            self.target_entropy, self.ent_coef_optim = None, None

    def replay(self) -> None:
        replay_start = time.time()
        self.actor_loss_list_once_replay = []
        self.critic_loss_list_once_replay = []
        self.ent_coef_loss_list_once_replay = []
        self.ent_coef_list_once_replay = []
        self.srl_loss_list_once_replay = []
        for _ in range(self.n_updates):
            # I: SRL model update
            start = time.time()
            data = self.buffer.sample(self.srl_batch_size)
            flatten = nn.Flatten(start_dim=1)
            predictive_transition_prior = self.srl_model.f_pred(flatten(data.observations), data.actions)
            assert predictive_transition_prior.shape == flatten(data.next_observations).shape
            srl_loss = nn.functional.mse_loss(predictive_transition_prior, flatten(data.next_observations))
            self.srl_model_optim.zero_grad()
            srl_loss.backward()
            self.srl_model_optim.step()
            # copy ofe net to actor,critic. not target net. SRL model training is actor's srl and critic's srl training
            self.actor.srl_model.load_state_dict(self.srl_model.state_dict())
            self.critic.srl_model.load_state_dict(self.srl_model.state_dict())
            end = time.time()
            print(f'srl once replay elapssed time: {end - start}')

            start = time.time()
            # get one batch samples:
            data = self.buffer.sample(self.rl_batch_size)
            # II: update ent_coef
            if self.auto_ent_coef:
                actions_pi, log_prob = self.actor.actions_log_prob(data.observations) # pi suffix, selected by current policy
                # assert actions_pi.shape==(self.batch_size, self.actor.action_dim)
                # assert log_prob.shape==(self.batch_size, 1)
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optim.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optim.step()
            else:
                ent_coef = torch.exp(self.log_ent_coef.detach())
            end = time.time()
            print(f'ent_coef once replay elapssed time: {end - start}')

            # III: update critic
            start = time.time()
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.actions_log_prob(data.next_observations)
                next_q_values = torch.cat(self.critic_target(data.next_observations, next_actions), dim=1)
                # assert next_q_values.shape == (self.batch_size, 2)
                next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
                # assert next_q_values.shape == next_log_prob.shape == (self.batch_size, 1)
                next_q_values = next_q_values - ent_coef * next_log_prob
                targets = data.rewards + self.gamma * (1 - data.dones) * (next_q_values)
            current_q_values = self.critic(data.observations, data.actions)
            if self.if_prioritized:
                # q_values_pre_means = sum(current_q_values) / self.critic.n_critics
                # td_errors = q_values_pre_means - targets
                # assert q_values_pre_means.shape == targets.shape == td_errors.shape
                # self.buffer.update_priorities([*zip(data.sample_idx, abs(td_errors))])

                # 1.cal td errors
                td_errors = sum(q_values_pre for q_values_pre in current_q_values) / self.critic.n_critics - targets
                td_errors = td_errors.detach().numpy()
                # td_errors = q_value_pres1 - td_targets
                alpha = self.buffer.alpha
                # 2. update priorities
                self.buffer.update_priorities([*zip(data.sample_idx, (abs(td_errors)**alpha))])
                weights = (data.sample_probs / min(data.sample_probs))**(-self.buffer.beta())
                assert weights.requires_grad == False
                critic_loss1 = (((current_q_values[0] - targets) ** 2) * (weights / 2)).mean()
                critic_loss2 = (((current_q_values[1] - targets) ** 2) * (weights / 2)).mean()
                critic_loss = (critic_loss1 + critic_loss2) / 2
            else:
                critic_loss = 0.5 * sum(
                    nn.functional.mse_loss(current_q_value, targets) for current_q_value in current_q_values)
                # critic_loss = sum(
                #     nn.functional.mse_loss(current_q_value, targets) for current_q_value in current_q_values)
            self.critic.optim1.zero_grad()
            self.critic.optim2.zero_grad()
            critic_loss.backward()
            self.critic.optim1.step()
            self.critic.optim2.step()
            end = time.time()
            print(f'critic once replay elapssed time: {end - start}')

            # IV: update actor
            if self.logger.total_updates % self.policy_update_delay == 0:
                start = time.time()
                q_values_pi = torch.cat(self.critic(data.observations, actions_pi), dim=1)
                min_q_values_pi, _ = torch.min(q_values_pi, dim=1, keepdim=True)
                # assert min_q_values_pi.shape == log_prob.shape == (self.batch_size, 1)
                actor_loss = (ent_coef * log_prob - min_q_values_pi).mean()
                self.actor.optim.zero_grad()
                actor_loss.backward()
                self.actor.optim.step()
                self.actor_loss_list_once_replay.append(actor_loss.detach().numpy())
                end = time.time()
                print(f'actor once replay elapssed time: {end - start}')
            if self.logger.total_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            self.critic_loss_list_once_replay.append(critic_loss.detach().numpy())
            self.ent_coef_list_once_replay.append(ent_coef.numpy())
            self.ent_coef_loss_list_once_replay.append(ent_coef_loss.detach().numpy())
            self.srl_loss_list_once_replay.append(srl_loss.detach().numpy())
            self.logger.total_updates_plus()
        replay_end = time.time()
        print(f'once replay elapsted time: {replay_end - replay_start}')

    def train(self):
        self.logger = episode_logger()
        for ep in range(self.episodes):
            self.logger.reset()
            done = False
            s = self.env_train.reset()
            while not done:
                s_tensor = torch.tensor(s, dtype=torch.float32).reshape((1,) + s.shape)
                a = self.actor.get_actions(s_tensor, False).detach().numpy().reshape(-1, )   # np.ndarray
                a_env = a * (self.action_range[1] - self.action_range[0]) / 2.0 +  \
                            (self.action_range[1] + self.action_range[0]) / 2.0
                # a: sqashed action; a_env:required by environment step.
                s_, r, done, _ = self.env_train.step(a_env)   # list,scalar,bool,empty_dict
                self.buffer.add(s, a, r, s_, done)   # np.ndarray,np.ndarray,scale,list,bool
                s = s_
                self.logger.timesteps_plus()

                if self.logger.total_timesteps > self.train_start and self.logger.total_timesteps > self.rl_batch_size:
                    self.replay()
                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r,
                                       asset=self.env_train.asset_memory[-1],
                                       time=f'{year[2:]}_{month}_{day}',
                                       critic_loss=np.mean(self.critic_loss_list_once_replay),
                                       srl_loss=np.mean(self.srl_loss_list_once_replay),
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
                                       srl_loss=0,
                                       )
                if  self.logger.timesteps % self.print_interval == 0 and self.logger.timesteps > 0:
                    self.logger.show(policy_update_delay=self.policy_update_delay, interval=self.print_interval, n_updates=self.n_updates)
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
                      # 'actor_optim_state_dict': self.actor.optim.state_dict(),
                      # 'critic_optim_state_dict': self.critic.optim.state_dict()
                      }
        torch.save(checkpoint, self.filename + '.pth')
        self.logger.save(self.filename)

    def load(self, path):
        return self.actor.load_state_dict(torch.load(path)['actor_state_dict'])
