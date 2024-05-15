import numpy as np
import torch
import time
from collections import defaultdict
# from torch.utils.tensorboard import SummaryWriter


class episode_logger():
    # for off policy algorithms, like ddpg\td3\sac\
    def __init__(self):
        self.episode = 0
        self.timesteps = 0
        self.total_timesteps = 0
        self.start_time = time.time()
        self.record_dict = defaultdict(list)
        self._plot_record_dict = {}  # record total training process
        # self.writer = SummaryWriter('./runs')
        self.total_updates = 0

    def record(self,**kwargs):
        # record result in corresponding dict with the parameter name
        for key,value in kwargs.items():
            self.record_dict[key].append(value.detach().numpy() if torch.is_tensor(value) else value)

    def reset(self):
        # reset the record dict for each episode
        self.episode += 1
        print(f'******************************   Episode : {self.episode}  ********************************')
        self.record_dict = defaultdict(list)
        self.timesteps = 0

    def show(self, policy_update_delay=1, interval=100, n_updates=1):
        # print mean results
        print(f' ---------------------------------------')
        print(f'| Episode {self.episode}, Timesteps {self.timesteps}')
        print(f'| in the last {interval} timesteps :')
        for key in sorted(self.record_dict.keys()):
            name = key
            if isinstance(self.record_dict[key][0], str):
                continue
            else:
                value = self.record_dict[key]
                if name == 'actor_loss':
                    if n_updates >= policy_update_delay:
                        start_index = -interval
                    else:
                        start_index = -(interval * n_updates) // policy_update_delay
                    mean = format(np.mean(value[start_index:]), '.4f')
                elif name == 'asset':
                    mean = format(value[-1], '.4f')
                elif name == 'reward':
                    mean = format(np.mean(value[-interval:]), '.4f')
                else:
                    mean = format(np.mean(value[-interval * n_updates:]), '.4f')  # critic loss or ent coef loss
                print('| mean of ' + name + ' is ' + ' '*(14-len(name)) + '| ' + mean)
        print(f'| total_timesteps is        | {self.total_timesteps}')
        print(f'| total_updates is          | {self.total_updates}')
        print(f' ---------------------------------------')

    def episode_results_sample(self):
        # extract part of each episode's results, used for plotting
        # for example, one episode's length is 3800, each episode will have 3800 timesteps and corresponding values(except actor loss)
        # we extract these values at fixed internals, and save each episode's extraction values
        self._plot_record_dict[self.episode] = defaultdict(list)  # save this episode's extraction results
        extraction_interval = 30                                  # extraction interval
        for key, values in self.record_dict.items():
            if key[:3] == 'val' or key[:4] == 'test':
                values = np.array(values)
                self._plot_record_dict[self.episode][key].extend(values)
            elif key == 'actor_loss':
                extraction_timesteps = [i * extraction_interval for i in
                                        range(len(self.record_dict[key]) // extraction_interval)]
                values = np.array(values)
                self._plot_record_dict[self.episode][key].extend(values[extraction_timesteps])
            else:
                extraction_timesteps = [i * extraction_interval for i in
                                        range(len(self.record_dict[key]) // extraction_interval)]
                values = np.array(values)
                self._plot_record_dict[self.episode][key].extend(values[extraction_timesteps])

    def total_updates_plus(self):
        # training times
        self.total_updates += 1

    def timesteps_plus(self):
        # interact times with env
        self.timesteps += 1
        self.total_timesteps += 1

    def print_elapsed_time(self):
        self.end_time = time.time()
        print(f'elapsed time is {(self.end_time - self.start_time)//60} min')

    def save(self, name):
        print('save plot results')
        np.save('training_results'+ name + '.npy', self._plot_record_dict)


class rollout_logger():
    # for on-policy algorithms, like ppo\a2c
    def __init__(self):
        self.episode = 0
        self.timesteps = 0
        self.total_timesteps = 0
        self.start_time = time.time()
        self.record_dict = defaultdict(list)
        self._plot_record_dict = {}
        self.total_updates = 1
        self.have_inferenced = False

    def reset(self):
        # reset the record dict for each rollout
        self._episode_results_store()
        self.record_dict = defaultdict(list)

    def record(self,**kwargs):
        # record result in corresponding dict according to the parameter name
        for key,value in kwargs.items():
            self.record_dict[key].append(value.detach().numpy() if torch.is_tensor(value) else value)

    def show(self):
        # print mean results
        print(f' ---------------------------------------')
        print(f'| Episode {self.episode}, Timesteps {self.timesteps}, Total_Timesteps {self.total_timesteps}')
        print(f'| in the last rollout training process :')
        for key in sorted(self.record_dict.keys()):
            if isinstance(self.record_dict[key][0], str):
                continue
            else:
                name = key
                value = self.record_dict[key]
                if name == 'asset':
                    mean = mean = format(value[-1], '.4f')
                else:
                    mean = format(np.mean(value), '.4f')
                print('| mean of ' + name + ' is ' + ' '*(14-len(name)) + '| ' + mean)
        print(f' ---------------------------------------')

    def episode_start(self):
        self.have_inferenced = False
        self.episode += 1
        self._plot_record_dict[self.episode] = defaultdict(list)
        self.timesteps = 0
        print(f'******************************   Episode : {self.episode}  ********************************')

    def _episode_results_store(self):
        # store results of one roll out collection
        for key, value in self.record_dict.items():
            if key == 'asset':
                self._plot_record_dict[self.episode][key].extend([self.record_dict[key][-1]])
            else:
                self._plot_record_dict[self.episode][key].extend([np.mean(self.record_dict[key])])

    def print_elapsed_time(self):
        self.end_time = time.time()
        print(f'elapsed time is {(self.end_time - self.start_time)//60} min')

    def timesteps_plus(self):
        self.timesteps += 1
        self.total_timesteps += 1

    def total_updates_plus(self):
        # training times
        self.total_updates += 1

    def save(self, name):
        np.save('training_results'+ name + '.npy', self._plot_record_dict)
