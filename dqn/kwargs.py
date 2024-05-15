env_kwargs = {'stock_dim':1,              # *ACTION_DIM
              'hmax':100,
              'initial_amount':1000000,
              'num_stock_shares':None,       # [0]*ACTION_DIM
              'buy_cost_pct':None,           # [0.001]*ACTION_DIM
              'sell_cost_pct':None,          # [0.001]*ACTION_DIM
              'reward_scaling':1e-4,         # reward=reward*reward_scaling
              'state_space':None,            # *STATE_DIM
              'action_space':None,           # *ACTION_DIM
              'tech_indicator_list':['macd','boll_ub','boll_lb','rsi_30','cci_30','dx_30','close_30_sma','close_60_sma'],
              'turbulence_threshold':None,
              'risk_indicator_col':'turbulence',}

model_params_init_kwargs = {'aliase':'xavier_uniform',
                            'activation':'relu',
                            'a':0,
                            'seed':1,
                            'flag':False
                            }

policy_kwargs = {'net_arch': [64, 64],
                 'state_dim': None,
                 'action_dim': None,
                 'optim_aliase': 'adam',
                 'lr': 1e-4,
                 'model_params_init_kwargs': model_params_init_kwargs,
                 'epsilon_schedule_aliase': None,
                 'epsilon_schedule_kwargs': None,
                 }

epsilon_schedule_kwargs = {
    'start_time': 1,
    'end_time': None,
    'start_point': 1,
    'end_point': None,

}

agent_kwargs ={'env': None,
               'episodes': 10,
               'buffer_size': int(1e5),
               'batch_size': 100,
               'gamma': 0.99,
               'training_start': 100,
               'policy_kwargs': None,
               'target_update_interval': 100,
               'tau': 1.0,  # default 1 for hard update
               'filename': None,
               }

test_kwargs = {'train': None,
               'trade': None,
               'env_kwargs': None,
               'agent': None,   # agent is very important
               'risk_free': 0.03,
               'baseline_data_dir': 'E:/强化学习/强化学习代码/数据集/道指原始数据/DJI/',
               'TRAIN_START_DATE': None,
               'TRAIN_END_DATE': None,
               'TRADE_START_DATE': None,
               'TRADE_END_DATE': None,
               'if_actor': None,
               'if_action_discrete': None,
                }