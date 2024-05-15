# paper: Proximal Policy Optimization Algorithms
# model: Proximal Policy Optimization
# innovation: Clipped Surrogate Objective or Adaptive KL Penalty Coefficient, both of them is to simplify TRPO model.

import torch
import warnings
import numpy as np
from torch import nn
from finrl_myself.buffers.RollotBuffer import RolloutBuffer
from finrl_myself.ppo.policy import *
from finrl_myself.logger import rollout_logger
from finrl_myself.examine import validate, test


class PPO():

    def __init__(
            self,
            env_train=None,
            env_validation=None,
            env_test=None,
            total_timesteps: int = None,
            n_updates: int = None,
            buffer_size: int = None,
            n_rollout_steps: int = None, # equal to buffer size
            batch_size: int = None,
            lambda_coef: float = 0.95,
            gamma: float = 0.99,
            clip_range: float = 0.2,
            ent_coef: float = 0,
            value_coef: float = 0.5,
            policy_kwargs: dict = None,
            filename: str = None,
    ):

        self.env_train = env_train
        self.env_validation = env_validation
        self.env_test = env_test
        self.total_timesteps = total_timesteps
        self.n_updates = n_updates
        self.buffer_size = buffer_size
        self.n_rollout_steps = n_rollout_steps
        self.buffer = RolloutBuffer(buffer_size, batch_size, lambda_coef, gamma, self.env_train.observation_space.shape, self.env_train.action_dim)
        self.policy = D2RLPolicy(**policy_kwargs)   # no target net
        self.last_state = self.env_train.reset()  # ndarray
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.value_coef = value_coef
        self._last_episode_starts = True
        self.filename = filename


        if n_rollout_steps % batch_size > 0:
            untruncated_batches = n_rollout_steps // batch_size
            warnings.warn(
                f"You have specified a mini-batch size of {batch_size},"
                f" but because the `RolloutBuffer` is of size `n_rollout_steps = {n_rollout_steps}`,"
                f" after every {untruncated_batches} untruncated mini-batches,"
                f" there will be a truncated mini-batch of size {n_rollout_steps % batch_size}\n"
                f"We recommend using a `batch_size` that is a factor of `n_rollout_steps`.\n"
                f"Info: (n_rollout_steps={self.n_rollout_steps})"
            )

    def collect_rollout(self)->None:
        '''
        Collect a whole rollout data.
        When one episode ends but rollout is not complete
        Env will be reset.
        '''
        n_steps = 0
        self.buffer.reset()
        while n_steps < self.n_rollout_steps:
            last_state_tensor = torch.tensor(self.last_state, dtype=torch.float32).reshape((1,) + self.last_state.shape)
            with torch.no_grad():  # value is going to be used to compute return, which does not need gradient
                action, value, log_prob = self.policy(last_state_tensor)  # tensor,tensor,tensor log_prob is log(pi_ref(at|st))
                gaussian_action = self.policy.get_gaussian_actions()

            action = action.detach().numpy().reshape(-1,)  # ndarray
            gaussian_action = gaussian_action.detach().numpy().reshape(-1,)
            value = value.detach().numpy()
            log_prob = log_prob.detach().numpy()

            next_state, reward, done, _ = self.env_train.step(action)   # ndarray, float, bool, dict
            self.buffer.add(self.last_state,
                            action,
                            gaussian_action,
                            reward,
                            self._last_episode_starts,  # if self.last_state is the beginning of an episode
                            value,
                            log_prob)  # ndarray, ndarray, float, bool, ndarray, ndarray

            if done:
                self.last_state = self.env_train.reset()
                self._last_episode_starts = True
            else:
                self.last_state = next_state
                self._last_episode_starts = done

            n_steps += 1
            self.logger.record(reward=reward, asset=self.env_train.asset_memory[-1])
            self.logger.timesteps_plus()
            if done:
                self.logger.episode_start()

        with torch.no_grad(): # last_value also does not need gradient
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).reshape((1,) + next_state.shape)
            last_value = self.policy.predict_values(next_state_tensor)   # tensor
        self.buffer.compute_return_and_advantages(last_value.detach().numpy(), done)

    def replay(self):
        for _ in range(self.n_updates):
            self.rollout_data = self.buffer.get_rollout_samples()
            for data in self.rollout_data:
                states = data.states
                actions = data.actions
                gaussian_actions = data.gaussian_actions
                value_targets = data.returns
                log_prob_old = data.log_prob_old # pi_theta_old(a|s)
                advantages = data.advantages

                values, log_prob, entropy = self.policy.evaluate_actions(obs=states,
                                                                         actions=actions,
                                                                         gaussian_actions=gaussian_actions)

                # I: update critic
                # assert values.shape==value_targets.shape==(self.buffer.batch_size, 1)
                value_loss = nn.functional.mse_loss(value_targets, values)

                # # II: update actor
                ratio = torch.exp(log_prob - log_prob_old)
                # assert log_prob.shape == ratio.shape == (self.buffer.batch_size, 1)
                # assert entropy.shape == (self.env.action_dim, 1)
                policy_loss = -torch.min(torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages,
                                         ratio * advantages)
                policy_loss = policy_loss.mean()
                entropy_loss = -torch.mean(entropy)

                loss = policy_loss + self.ent_coef * entropy_loss + self.value_coef * value_loss
                self.policy.optim.zero_grad()
                loss.backward()
                self.policy.optim.step()
                self.logger.record(value_loss=value_loss, policy_loss=policy_loss, entropy_loss=entropy_loss, loss=loss)
                self.logger.total_updates_plus()

    def train(self):
        self.logger = rollout_logger()
        self.logger.episode_start()
        while self.logger.total_timesteps < self.total_timesteps:
            self.collect_rollout()
            self.replay()
            self.logger.show()
            if ((self.logger.episode + 1) % 1) == 0 and (not self.logger.have_inferenced):
                validate(self.env_validation, self.policy, self.logger)
                test(self.env_test, self.policy, self.logger)
                self.logger.have_inferenced = True
            self.logger.reset()
        self.logger.print_elapsed_time()

    def save(self, filename = None):
        if self.filename is None and filename is None:
            self.filename = 'test'
        elif filename is not None:
            self.filename = filename
        checkpoint = {
            'policy_state_dict':self.policy.state_dict(),
            'policy_optim_state_dict':self.policy.optim.state_dict()
        }
        torch.save(checkpoint, self.filename + '.pth')

        self.logger.save(self.filename)

    def load(self, path):
        return self.policy.load_state_dict(torch.load(path)['policy_state_dict'])
