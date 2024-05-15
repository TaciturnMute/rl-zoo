import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from finrl_myself.metrics import *
from finrl_myself.redq.redq_sac.critic import *
from finrl_myself.redq.redq_sac.actor import *
from finrl_myself.buffers.ReplayBuffer import ReplayBuffer
from finrl_myself.utils import polyak_update
from finrl_myself.logger import episode_logger
from finrl_myself.examine import validate, test


class REDQ():
    """
    Naive SAC: N = 2, M = 2
    REDQ: N > 2, M = 2
    MaxMin: N = M
    for above three variants, set q_target_mode to 'min' (default)
    Ensemble Average: set q_target_mode to 'ave'
    REM: set q_target_mode to 'rem'
    """
    def __init__(
            self,
            env_train,
            env_validation,
            env_test,
            episodes: int = 10,
            n_updates: int = 5,
            N: int = None,
            M: int = None,
            buffer_size: int = 100000,
            n_steps: int = None,
            if_prioritized: int = None,
            batch_size: int = 100,
            actor_aliase: str = None,
            critic_aliase: str = None,
            training_start: int = 100,
            target_update_interval: int = 1,
            policy_update_delay: int = 1,
            gamma: float = 0.99,
            tau: float = 0.005,
            filename: str = None,
            critic_kwargs: dict = None,
            actor_kwargs: dict = None,
            print_interval: int = 100,
            init_value: float = 1.0,
            auto_ent_coef: bool = True,   # if used auto-adapting alpha
            target_entropy_mode='auto',   # other target entropy alternative can be used
            ent_coef_lr: float = 3e-4,
            q_target_mode='redq',
    ):
        # from here get specific actor and critic
        actor_aliases: dict = {
            "SequentialMlp": Actor,
        }
        critic_aliases: dict = {
            "SequentialMlp": Critic,
            "DualingFore": Critic_DualingFore,
        }
        # set up networks
        self.env_train = env_train
        self.env_validation = env_validation
        self.env_test = env_test
        self.N = N
        self.M = M
        self.episodes = episodes
        self.n_steps = n_steps
        self.if_prioritized = if_prioritized
        self.training_start = training_start
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.filename = filename
        self.n_updates = n_updates
        self.print_interval = print_interval
        self.gamma = gamma
        self.batch_size = batch_size
        self.q_target_mode = q_target_mode
        self.policy_update_delay = policy_update_delay
        self.Actor = actor_aliases[actor_aliase]
        self.actor = self.Actor(**actor_kwargs)
        self.Critic = critic_aliases[critic_aliase]
        self.critic = self.Critic(**critic_kwargs)
        self.critic_target = self.Critic(**critic_kwargs)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.buffer = ReplayBuffer(buffer_capacity=buffer_size,
                                   batch_size=batch_size,
                                   n_steps=n_steps,
                                   gamma=gamma,
                                   if_prioritized=if_prioritized)
        # set up adaptive entropy (SAC adaptive)
        self.auto_ent_coef = auto_ent_coef
        if self.auto_ent_coef:
            if target_entropy_mode == 'auto':
                self.target_entropy = -np.prod(self.env_train.action_space.shape).astype(np.float32)  # why  # float
            # if target_entropy_mode == 'mbpo':
            #     self.target_entropy = mbpo_target_entropy_dict[env_name]
            self.log_ent_coef = torch.log(torch.ones(1) * init_value).requires_grad_(True) # torch.float
            self.ent_coef_optim = optim.Adam([self.log_ent_coef], lr=ent_coef_lr)
        else:
            self.log_ent_coef = torch.log(torch.ones(1) * init_value)  # torch.float
            self.target_entropy, self.ent_coef_optim = None, None
        assert self.N == self.critic.N

    def _cal_target(self, rewards, next_states, dones, M_indexs, ent_coef):
        with torch.no_grad():
            if self.q_target_mode == 'redq':
                """Q target is min of a subset of Q values"""
                next_actions, next_log_prob = self.actor.actions_log_prob(next_states)  # 一列
                next_q_values_list = self.critic_target.get_M_values(M_indexs, next_states, next_actions)  # 多列
                next_q_values, _ = torch.min(torch.cat(next_q_values_list, 1), dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob
                target = rewards + self.gamma * (1 - dones) * next_q_values

            if self.q_target_mode == 'ave':
                """Q target is average of a subset of Q values"""
                next_actions, next_log_prob = self.actor.actions_log_prob(next_states)
                M_indexs = list(range(self.N))
                next_q_values_list = self.critic_target.get_M_values(M_indexs, next_states, next_actions)
                next_q_values = torch.cat(next_q_values_list, 1).mean(dim=1).reshape(-1, 1)  # average of all qf
                next_q_values = next_q_values - ent_coef * next_log_prob
                target = rewards + self.gamma * (1 - dones) * next_q_values

            if self.q_target_mode == 'weighted':
                next_actions, next_log_prob = self.actor.actions_log_prob(next_states)
                next_q_values_list = self.critic_target.get_M_values(M_indexs, next_states, next_actions)
                # concat and sorted, result shape is (batch_size, N). each row is sorted values
                sorted_values_of_all_samples = torch.cat(next_q_values_list, axis=1).sort()[0]
                weights = [math.factorial(self.N - i) / \
                           (math.factorial(self.M - 1) * math.factorial(self.N - i - self.M + 1)) \
                           for i in range(1, self.N - self.M + 2)] + [0] * (self.M - 1)
                weighted_sum_values = sorted_values_of_all_samples.mul_(torch.tensor(weights))
                N_select_M = math.factorial(self.N) / (math.factorial(self.M) * math.factorial(self.N - self.M))
                next_q_values = (weighted_sum_values.sum(axis=1) / N_select_M).reshape(-1, 1)
                next_q_values = next_q_values - ent_coef * next_log_prob
                target = rewards + self.gamma * (1 - dones) * next_q_values

        return target

    def replay(self):
        self.actor_loss_list_once_replay = []
        self.critic_loss_list_once_replay = []
        self.ent_coef_loss_list_once_replay = []
        self.ent_coef_list_once_replay = []
        for _ in range(self.n_updates):
            # get one batch samples:
            data = self.buffer.sample()
            # I: ent_coef loss
            if self.auto_ent_coef:
                actions_pi, log_prob = self.actor.actions_log_prob(data.observations)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                self.ent_coef_optim.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optim.step()
                ent_coef = torch.exp(self.log_ent_coef.detach())
            else:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = torch.tensor(0)
            # II: critic loss
            M = self._get_probabilistic_num_min(self.M)
            M_indexs = np.random.choice(self.N, M, replace=False).tolist()  # [0,N-1]
            # calculate TD target
            targets = self._cal_target(data.rewards, data.next_observations, data.dones, M_indexs, ent_coef)
            current_q_values_list = self.critic(data.observations, data.actions)
            targets = targets.expand((-1, self.N)) if targets.shape[1] == 1 else targets
            if self.if_prioritized:
                td_errors = sum(q_values_pre for q_values_pre in current_q_values_list) / self.N - targets
                td_errors = td_errors.detach().numpy()
                alpha = self.buffer.alpha
                self.buffer.update_priorities([*zip(data.sample_idx, (abs(td_errors)**alpha))])
                weights = (data.sample_probs / min(data.sample_probs))**(-self.buffer.beta())
                assert weights.requires_grad == False
                weights = weights.expand((-1, self.N))
                q_values_pre_all = torch.cat(current_q_values_list, dim=1)
                assert weights.shape == targets.shape == q_values_pre_all.shape
                critic_loss = (((q_values_pre_all - targets)**2) * (weights / 2)).mean()
            else:
                critic_loss = nn.functional.mse_loss(torch.cat(current_q_values_list, dim=1), targets)
            for index in range(self.N):
                self.critic.optim_list[index].zero_grad()
            critic_loss.backward()
            for index in range(self.N):
                self.critic.optim_list[index].step()

            # III: actor loss
            if self.logger.total_updates % self.policy_update_delay == 0:
                q_values_pi = torch.cat(self.critic(data.observations, actions_pi), dim=1)      # shape: (batch_size, N)
                mean_q_values_pi = torch.mean(q_values_pi, dim=1, keepdim=True)      # shape: (batch_size, 1)
                actor_loss = (ent_coef * log_prob - mean_q_values_pi).mean()
                self.actor.optim.zero_grad()
                actor_loss.backward()
                self.actor.optim.step()
                self.actor_loss_list_once_replay.append(actor_loss.detach().numpy())
            self.critic_loss_list_once_replay.append(critic_loss.detach().numpy())
            self.ent_coef_list_once_replay.append(ent_coef.numpy())
            self.ent_coef_loss_list_once_replay.append(ent_coef_loss.detach().numpy())
            if self.logger.total_updates % self.target_update_interval == 0:
                for i in range(self.N):
                    polyak_update(self.critic.qf_list[i].parameters(), self.critic_target.qf_list[i].parameters(), self.tau)
            self.logger.total_updates_plus()

    def train(self):
        self.logger = episode_logger()
        for ep in range(self.episodes):
            self.logger.reset()
            done = False
            s = self.env_train.reset()
            while not done:
                s_tensor = torch.tensor(s, dtype=torch.float32).reshape((1,) + s.shape)
                a = self.actor.get_actions(s_tensor).detach().numpy().reshape(-1, )
                s_, r, done, _ = self.env_train.step(a)
                self.buffer.add(s, a, r, s_, done)
                s = s_
                self.logger.timesteps_plus()
                # train
                if self.logger.total_timesteps > self.training_start and self.logger.total_timesteps > self.batch_size:
                    self.replay()
                    date_list = self.env_train.date.split('-')
                    year, month, day = date_list[0], date_list[1], date_list[2]
                    self.logger.record(reward=r,
                                       asset=self.env_train.asset_memory[-1],
                                       time=f'{year[2:]}_{month}_{day}',
                                       critic_loss=np.mean(self.critic_loss_list_once_replay),
                                       ent_coef_loss=np.mean(self.ent_coef_loss_list_once_replay),
                                       ent_coef=np.mean(self.ent_coef_list_once_replay)
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
                if self.logger.timesteps % self.print_interval == 0 and self.logger.timesteps > 0:
                    self.logger.show(
                        policy_update_delay=self.policy_update_delay,
                        interval=self.print_interval,
                        n_updates=self.n_updates)
            validate(self.env_validation, self.actor, self.logger)
            test(self.env_test, self.actor, self.logger)
            self.logger.episode_results_sample()
        self.logger.print_elapsed_time()

    def _get_probabilistic_num_min(self, num_mins):
        # allows the number of min to be a float
        floored_num_mins = np.floor(num_mins)
        if num_mins - floored_num_mins > 0.001:
            prob_for_higher_value = num_mins - floored_num_mins
            if np.random.uniform(0, 1) < prob_for_higher_value:
                return int(floored_num_mins+1)
            else:
                return int(floored_num_mins)
        else:
            return num_mins

    def save(self, filename = None):
        if self.filename is None and filename is None:
            self.filename = 'test'
        elif filename is not None:
            self.filename = filename
        checkpoint = {'actor_state_dict': self.actor.state_dict(),
                      # 'critic_state_dict': self.critic.state_dict(),
                      # 'actor_optim_state_dict': self.actor.optim.state_dict(),
                      # 'critic_optim_state_dict': self.critic.optim.state_dict()
                      }
        torch.save(checkpoint, self.filename + '.pth')

        self.logger.save(self.filename)

    def load(self, path):
        return self.actor.load_state_dict(torch.load(path)['actor_state_dict'])
