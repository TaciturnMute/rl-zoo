class ReplayBuffer():

    def __init__(
            self,
            keys: list = ['observation', 'action', 'reward', 'next_observation', 'done'],
            buffer_capacity: int = None,
            batch_size: int = None,

    ):
        self.keys = keys
        self.buffer = collections.deque(maxlen=buffer_capacity)
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.ReplayBufferSamples = namedtuple('Transition', self.keys)

    def add(self,
            state: np.ndarray,  # 没有batch维度
            action: np.ndarray, # 没有batch维度
            reward: float,   # 标量
            next_state: list,  # 没有batch维度
            done: bool) -> None:
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        batch_samples = random.sample(self.buffer, self.batch_size)
        observations, actions, rewards, next_observations, dones = map(np.asarray, zip(*batch_samples))  # 转化成np.ndarray，方便
        dones = np.where(dones, 1, 0)

        observations = CompleteShape(observations)
        next_observations = CompleteShape(next_observations)
        rewards = CompleteShape(rewards)
        dones = CompleteShape(dones)

        observations = torch.tensor(observations, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_observations = torch.tensor(next_observations, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        data = (observations, actions, rewards, next_observations, dones)
        replay_data = self.ReplayBufferSamples(*tuple(data))

        return replay_data


class ReplayBufferZero():

    def __init__(self, buffer_size: int, batch_size: int):
        # self.observation_buffer = deque(maxlen=buffer_size)
        # self.action_buffer = deque(maxlen=buffer_size)
        # self.reward_buffer = deque(maxlen=buffer_size)
        # self.done_buffer = deque(maxlen=buffer_size)
        self.buffer = deque(maxlen=buffer_size * 4)
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def add(self,
            observation: np.ndarray,  # 没有batch维度
            action: np.ndarray, # 没有batch维度
            reward: float,   # 标量
            next_observation: list,  # 没有batch维度
            done: bool) -> None:

        # self.observation_buffer.append(observation)
        # self.action_buffer.append(action)
        # self.reward_buffer.append(reward)
        # self.done_buffer.append(done)
        self.buffer.append(observation)
        self.buffer.append(action)
        self.buffer.append(reward)
        self.buffer.append(done)

    def sample(self) -> Tuple[torch.Tensor, ...]:

        assert len(self.buffer) % 4 == 0
        batch_indexs = np.random.randint(0, len(self.buffer) / 4, self.batch_size)
        batch_observations = []
        batch_actions = []
        batch_rewards = []
        batch_next_observations = []
        batch_dones = []
        for idx in batch_indexs:
            batch_observations.append(self.buffer[idx * 4])
            batch_actions.append(self.buffer[idx * 4 + 1])
            batch_rewards.append(self.buffer[idx * 4 + 2])
            done = self.buffer[idx * 4 + 3]
            batch_dones.append(done)
            if not done:
                batch_next_observations.append(self.buffer[idx * 4 + 4])
            else:
                batch_next_observations.append(self.buffer[idx * 4])

        batch_observations, batch_actions, batch_rewards, batch_next_observations, batch_dones = map(
            np.asarray, [batch_observations, batch_actions,  batch_rewards, batch_next_observations, batch_dones]
        )

        batch_dones = np.where(batch_dones, 1, 0)

        batch_observations, batch_actions, batch_rewards, batch_next_observations, batch_dones = map(
            CompleteShape, [batch_observations, batch_actions,  batch_rewards, batch_next_observations, batch_dones]
        )


        # states = CompleteShape(batch_observations)
        # next_states = CompleteShape(batch_next_observations)
        # rewards = CompleteShape(batch_rewards)
        # dones = CompleteShape(batch_dones)


        # batch_samples = random.sample(self.buffer, self.batch_size)
        # states, actions, rewards, next_states, dones = map(np.asarray, zip(*batch_samples))  # 转化成np.ndarray，方便
        # dones = np.where(dones, 1, 0)
        #
        # states = CompleteShape(states)
        # next_states = CompleteShape(next_states)
        # rewards = CompleteShape(rewards)
        # dones = CompleteShape(dones)
        #
        batch_observations = torch.tensor(batch_observations, dtype=torch.float32)
        batch_actions = torch.tensor(batch_actions, dtype=torch.float32)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        batch_next_observations = torch.tensor(batch_next_observations, dtype=torch.float32)
        batch_dones = torch.tensor(batch_dones, dtype=torch.float32)

        return (batch_observations, batch_actions, batch_rewards, batch_next_observations, batch_dones)

    @property
    def size(self):
        return self.buffer_size, self.batch_size

    def get_buffer(self):
        return self.buffer