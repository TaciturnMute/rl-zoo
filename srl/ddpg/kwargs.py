env_trade_kwargs = {
    'stock_dim': None,
    'hmax': 100,
    'initial_amount': 1000000,
    'num_stock_shares': None,    # [0]*stock_dim
    'buy_cost_pct': None,    # [0.001]*ACTION_DIM
    'sell_cost_pct': None,      # [0.001]*ACTION_DIM
    'reward_scaling': 1e-4,         # reward=reward*reward_scaling
    'state_space': None,
    'action_space': None,
    'tech_indicator_list': ['macd','boll_ub','boll_lb','rsi_30','cci_30','dx_30','close_30_sma','close_60_sma'],
    'turbulence_threshold': None,
    'risk_indicator_col': 'turbulence',
    'reward_aliase': 'asset_diff',
}

env_trade_series_kwargs = {
    'history':None,
    'stock_dim': None,
    'hmax': 100,
    'initial_amount': 1000000,
    'num_stock_shares': None,    # [0]*stock_dim
    'buy_cost_pct': None,    # [0.001]*ACTION_DIM
    'sell_cost_pct': None,      # [0.001]*ACTION_DIM
    'reward_scaling': 1e-4,         # reward=reward*reward_scaling
    'state_space': None,
    'action_space': None,
    'tech_indicator_list': ['macd','boll_ub','boll_lb','rsi_30','cci_30','dx_30','close_30_sma','close_60_sma'],
    'turbulence_threshold': None,
    'risk_indicator_col': 'turbulence',
    'reward_aliase': 'asset_diff',
}

env_portfolio_kwargs = {
    'stock_dim': None,    # len(df.tic.unique())
    'hmax': 100,
    'initial_amount': 1000000,
    'buy_cost_pct': None,
    'sell_cost_pct': None,
    'reward_scaling': 1e-4,
    'state_space': None,   # stock_dim
    'action_space': None,  # stock_dim
    'tech_indicator_list': ['macd', 'rsi_30', 'cci_30', 'dx_30'],
    'turbulence_threshold': None,
    'lookback': 252,
    'reward_aliase': 'asset_diff',
}

env_portfolio_series_kwargs = {
    'history':None,
    'stock_dim': None,    # len(df.tic.unique())
    'hmax': 100,
    'initial_amount': 1000000,
    'buy_cost_pct': None,
    'sell_cost_pct': None,
    'reward_scaling': 1e-4,
    'state_space': None,   # stock_dim
    'action_space': None,  # stock_dim
    'tech_indicator_list': ['macd', 'rsi_30', 'cci_30', 'dx_30'],
    'turbulence_threshold': None,
    'lookback': 252,
    'reward_aliase': 'asset_diff',
}

model_params_init_kwargs = {
    'aliase': 'xavier_uniform',
    'activation': 'relu',
    'a': 0,
    'seed': 1,
    'flag': False
}

# used for mlp structure actor
actor_kwargs = {
    'net_arch': None,
    'srl_net_arch': None,
    'state_dim': None,
    'action_dim': None,
    'optim_aliase': 'adam',
    'lr': 1e-3,
    'model_params_init_kwargs': model_params_init_kwargs,
    'feature_extractor_aliase': 'flatten',
    'feature_extractor_kwargs': None,
    'dropout': 0,
    'activation_fn': None,
}

# used for mlp structure critic
critic_kwargs = {
    'net_arch': None,    # [400, 300] / [{'vf':[64,32], 'qf':[32]},16,16]
    'srl_net_arch':None,
    'state_dim': None,
    'action_dim': None,
    'model_params_init_kwargs': model_params_init_kwargs,
    'optim_aliase': 'adam',
    'lr':1e-3,
    'feature_extractor_aliase': 'flatten',
    'feature_extractor_kwargs': None,
    'dropout': 0,
    'activation_fn':None,
}

ou_noise_kwargs = {
    'mu': None,  # np.array([0]*ACTION_DIM)
    'sigma': 0.05,
    'theta': 0.15,
    'dt': 0.1,
    'x0': None,
    'seed': 1,
    'randomness': True,
}

normal_noise_kwargs = {
    'loc': None,   # np.array([0]*ACTION_DIM)
    'std': 0.05,
    'seed': 10,
    'randomness': True,
}

smooth_noise_kwargs = {
    'loc': None,  # np.array([0]*ACTION_DIM)
    'std': 0.05,
    'seed': 20,
    'randomness': True,
    'clip': 0.1,
    'batch_size': None,
}

agent_kwargs = {
    'env_train':None,
    'env_validation':None,
    'env_test':None,
    'buffer_size': int(1e5),
    'rl_batch_size': 100,
    'srl_batch_size': 100,
    'episodes': 10,
    'tau': 0.005,
    'gamma': 0.95,
    'training_start': 100,
    'n_updates':1,
    'srl_lr':None,
    'n_steps':1,
    'if_prioritized':False,
    'noise_aliase': 'ou',
    'noise_kwargs': None,
    'smooth_noise_aliase': None,
    'smooth_noise_kwargs': None,
    'if_smooth_noise': False,
    'print_interval': 100,
    'filename': None,
    'critic_kwargs': None,
    'actor_kwargs': None,
}

test_kwargs = {
    'env_train':None,
    'env_test':None,
    'agent': None,   # agent is very important
    'risk_free': 0.03,
    'baseline_data_dir': 'E:/强化学习/强化学习代码/数据集/DOW原始数据/DJI/',
    'TRAIN_START_DATE': None,
    'TRAIN_END_DATE': None,
    'TEST_START_DATE': None,
    'TEST_END_DATE': None,
    'if_actor': None,
}


feature_extractors_kwargs = {
    'flatten': {
        'observation_space': None,
    },

}
