# select best SRL model net_arch with auxiliary socre
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader,Dataset,TensorDataset
from tqdm import tqdm
import time

class SRL_Net_Select():

    def __init__(
            self,
            agent,
            env,
            srl_model,
            lr,
            data_collection_size,
            epochs,
            batch_size,
            if_agent_random_policy,
            if_noise,
    ):
        '''
        :param env: provide transitions
        :param srl_model: training model
        :param lr: learning rate
        :param data_collection_size: dataset size
        :param epochs: supervised learning training epochs
        :param batch_size: mini-batch size of sampling for supervised learning
        '''
        self.agent = agent
        self.env = env
        self.srl_model = srl_model
        self.optim = torch.optim.Adam(params=srl_model.parameters(), lr=lr)
        self.data_collection_size = data_collection_size
        self.epochs = epochs
        self.env.batch_size = batch_size
        self.batch_size = batch_size
        self.if_agent_random_policy = if_agent_random_policy
        self.if_noise = if_noise
        self.total_updates = 0
        self.flatten = nn.Flatten(start_dim=1)
        self.loss_list = []


    def _dataloader(self):
        count = 0
        while count < self.data_collection_size:
            s = self.env.reset()
            done = False
            observation_dataset = []
            action_dataset = []
            next_observation_dataset = []
            while not done:
                s_tensor = torch.tensor(s, dtype=torch.float32).reshape((1,) + s.shape)
                if self.if_agent_random_policy:
                    a = self.agent.actor.get_actions(s_tensor, False).detach().numpy().reshape(-1, )  # (action_dim,)
                    if self.if_noise:
                        a = np.clip(a + self.agent.noise.__call__(), self.agent.action_space_range[0], self.agent.action_space_range[1])
                else:
                    a = self.env.action_space.sample()
                s_, r, done, _ = self.env.step(a)  # ndarray,float,bool,dict
                observation_dataset.append(list(s))
                action_dataset.append(list(a))
                next_observation_dataset.append(list(s_))
                count += 1
                if count % 1000 == 0 and count >= 1000:
                    print(f'have collected { count} data ')
        tensor_dataset = TensorDataset(torch.tensor(np.array(observation_dataset), dtype=torch.float32),
                                       torch.tensor(np.array(action_dataset), dtype=torch.float32),
                                       torch.tensor(np.array(next_observation_dataset), dtype=torch.float32)
                                       )
        collate_fn = lambda x: [torch.cat([x[i][j].unsqueeze(0) for i in range(len(x))]) for j in range(len(x[0]))]
        dataloader = DataLoader(dataset=tensor_dataset,
                                 batch_size=self.batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 sampler=None,
                                 batch_sampler=None,
                                 num_workers=0,
                                 collate_fn=collate_fn,
                                 )
        return dataloader

    def train(self):
        print('preparing dataloader...')
        self.dataloader = self._dataloader()
        print('training start!')
        start = time.time()
        for _ in tqdm(range(self.epochs)):
            loss_list = []
            for batch_data in self.dataloader:
                batch_observations = batch_data[0]
                batch_actions = batch_data[1]
                batch_next_observations = batch_data[2]
                batch_observations = self.flatten(batch_observations)
                batch_next_observations = self.flatten(batch_next_observations)
                predictive_transition_prior = self.srl_model.f_pred(batch_observations, batch_actions)
                assert predictive_transition_prior.shape == batch_next_observations.shape
                loss = nn.functional.mse_loss(predictive_transition_prior, batch_next_observations)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                loss_list.append(loss.detach().numpy())
            print(f'epoch {_} loss: {np.mean(loss_list)}')
            self.loss_list.append(np.mean(loss_list))
        end = time.time()
        print(f'elasped time: {end - start}')

