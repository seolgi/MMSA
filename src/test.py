from MMSA import MMSA_run
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='misa', choices=['misa', 'self_mm', 'mmim', 'cenet', 'almt', 'tfr_net'])
parser.add_argument('--dataset_name', type=str, default='mosi')
parser.add_argument('--seeds', type=list, default=[1111, 1112, 1113])
parser.add_argument('--test_seeds', type=list, default=[1111, 1112, 1113])
parser.add_argument('--gpu_ids', type=list, default=[7])
args = parser.parse_args()
print("args: ", args)

MMSA_run(args.model_name, args.dataset_name, 
         seeds=args.seeds, 
         test_seeds=args.test_seeds,
         gpu_ids=args.gpu_ids,
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