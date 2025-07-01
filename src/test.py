from MMSA import MMSA_run



MMSA_run('misa', 'mosi', 
         seeds=[1111], 
         test_seeds=[1111],
         gpu_ids=[7],
         res_save_dir='res/results',
         model_save_dir='res/saved_models',
         log_dir='res/logs')


"""
# run LMF on MOSI with default hyper parameters
MMSA_run('lmf', 'mosi', seeds=[1111, 1112, 1113], gpu_ids=[0])

# tune Self_mm on MOSEI with default hyper parameter range
MMSA_run('self_mm', 'mosei', seeds=[1111], gpu_ids=[1])

# run TFN on SIMS with altered config
config = get_config_regression('tfn', 'mosi')
config['post_fusion_dim'] = 32
config['featurePath'] = '~/feature.pkl'
MMSA_run('tfn', 'mosi', config=config, seeds=[1111])

# run MTFN on SIMS with custom config file
MMSA_run('mtfn', 'sims', config_file='./config.json')
"""