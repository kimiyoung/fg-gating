import train
import test
import argparse
import os
import numpy as np
import random
import lasagne

# NAME = 'matrix_gate_concat_fix_s81'
NAME = 'MODEL'

# parse arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', dest='model', type=str, default='GAReaderMatrix',
        help='base model - (GAReader || GAReaderpp || StanfordAR || DeepASReader)')
parser.add_argument('--mode', dest='mode', type=int, default=0, # ==============================
        help='run mode - (0-train+test, 1-train only, 2-test only, 3-val only)')
parser.add_argument('--regularizer', dest='regularizer', type=str, default='l2',
        help='l2 or l1 norm for regularizing word embeddings')
parser.add_argument('--lambda', dest='lambda', type=float, default=0.,
        help='weight of regularization')
parser.add_argument('--nhidden', dest='nhidden', type=int, default=128,
        help='GRU hidden state size')
parser.add_argument('--char_dim', dest='char_dim', type=int, default=25,
        help='Size of char embeddings (0 to turn off). Char GRU hidden size = 2*char_dim.')
parser.add_argument('--nlayers', dest='nlayers', type=int, default=3,
        help='Number of reader layers')
parser.add_argument('--dropout', dest='dropout', type=float, default=0.3, # ============================== 0.4 for cbt, 0.3 for wdw
        help='Dropout rate')
parser.add_argument('--word2vec', dest='word2vec', type=str, default='word2vec/word2vec_glove.txt',
        help='File with word embeddings. Should have header with number and size of embeddings.')

parser.add_argument('--dataset', dest='dataset', type=str, default='wdw', # ==============================
        help='Dataset - (cnn/questions || dailymail/questions || cbtcn || cbtne || wdw || wdw_relaxed)')
parser.add_argument('--train_emb', dest='train_emb', type=int, default=0, # ===============================
        help='Tune word embeddings - (0-No, 1-Yes)')

parser.add_argument('--subsample', dest='subsample', type=int, default=-1,
        help='Sample window size around candidates. (-1-no sampling)')
parser.add_argument('--seed', dest='seed', type=int, default=61, # ===============================
        help='Seed for different experiments with same settings')
parser.add_argument('--use_feat', dest='use_feat', type=int, default=1,
        help='Use token_in_query feature - (0-no, 1-yes)')
parser.add_argument('--train_cut', dest='train_cut', type=float, default=1.0,
        help='Cut training data size by factor (default - no cut)')
parser.add_argument('--gating_fn', dest='gating_fn', type=str, default='T.mul',
        help='Gating function (T.mul || Tsum || Tconcat)')
args = parser.parse_args()
params=vars(args)

np.random.seed(params['seed'])
random.seed(params['seed'])
lasagne.random.set_rng(np.random)

# save directory
w2v_filename = params['word2vec'].split('/')[-1].split('.')[0] if params['word2vec'] else 'None'
save_path = ('experiments/'+params['model']+'/'+params['dataset'].split('/')[0]+
        '/reg%s'%params['regularizer']+
        '%.3f'%params['lambda']+'_nhid%d'%params['nhidden']+'_nlayers%d'%params['nlayers']+
        '_dropout%.1f'%params['dropout']+'_%s'%w2v_filename+'_chardim%d'%params['char_dim']+
        '_train%d'%params['train_emb']+'_subsample%d'%params['subsample']+
        '_seed%d'%params['seed']+'_use-feat%d'%params['use_feat']+
        '_traincut%.1f'%params['train_cut']+'_gf%s'%params['gating_fn']+'{}/'.format(NAME))
if not os.path.exists(save_path): os.makedirs(save_path)

# train
if params['mode']<2:
    train.main(save_path, params)

# test
if params['mode']==0 or params['mode']==2:
    test.main(save_path, params)
elif params['mode']==3:
    test.main(save_path, params, mode='validation')
