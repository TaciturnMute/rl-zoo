env_trade_kwargs = {
    'stock_dim': None,
    'hmax': 100,
    'initial_amount': 1000000,
    'num_stock_shares': None,  # [0]*stock_dim
    'buy_cost_pct': None,   # [0.001]*stock_dim
    'sell_cost_pct': None,  # [0.001]*stock_dim
    'reward_scaling': 1e-4,         # reward=reward*reward_scaling
    'state_space': None,    # 291
    'action_space': None,  # stock_dim
    'tech_indicator_list': ['macd','boll_ub','boll_lb','rsi_30','cci_30','dx_30','close_30_sma','close_60_sma'], #环境真正使用的
    'turbulence_threshold': None,   # 测试的时候使用
    'risk_indicator_col': 'turbulence',   # 测试的时候可以选择turbulence或者vix
    'reward_aliase':'asset_diff',
}

env_portfolio_kwargs = {
    'stock_dim': None,    # len(df.tic.unique())
    'hmax': 100,
    'initial_amount': 1000000,
    'buy_cost_pct': None,     # [0.001]*stock_dim
    'sell_cost_pct': None,    # [0.001]*stock_dim
    'reward_scaling': 1e-4,
    'state_space': None,   # stock_dim
    'action_space': None,  # stock_dim
    'tech_indicator_list': ['macd', 'rsi_30', 'cci_30', 'dx_30'],
    'turbulence_threshold': None,
    'lookback': 252,
    'reward_aliase': 'asset_diff',
}

model_params_init_kwargs = {
    'aliase':'xavier_uniform',
    'activation':'relu',
    'a':0,
    'seed':1,
    'flag':False
}

policy_kwargs = {
    'net_arch':[{'pi':[64,64],'vf':[64,64]}],
    'state_dim':None,
    'action_dim':None,
    'activation_fn':None,
    'feature_extractor_aliase': 'flatten',
    'feature_extractor_kwargs': None,
    'log_std_init':1.0,
    'optim_aliase':'adam',
    'lr':1e-2,
    'model_params_init_kwargs':model_params_init_kwargs
}

agent_kwargs = {
    'env_train':None,
    'env_validation':None,
    'env_test':None,
    'total_timesteps':None,
    'buffer_size':200,
    'n_rollout_steps':200,
    'n_updates':1,
    'batch_size':50,
    'lambda_coef':0.95,
    'gamma':0.99,
    'clip_range':0.2,
    'ent_coef':0,
    'value_coef':0.5,
    'filename':None,
    'policy_kwargs': policy_kwargs,
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
    'flatten': {'observation_space': None,},
}