import argparse
import numpy as np
import os

from train import MLPTrainer
from utils import *

def main(config):
    val_loss, test_loss = [], []
    val_folds = np.arange(config['num_folds'])
    test_folds = np.roll(val_folds, 1)
    for val_fold, test_fold in zip(val_folds, test_folds):
        config['val_fold'] = val_fold
        config['test_fold'] = test_fold
        trainer = MLPTrainer(config)
        val_loss.append(trainer.train())
        test_loss.append(trainer.test())
        print(val_loss)
        print(test_loss)
    dpath = 'results/mlp_{}_{}_{}_{}'.format(
        config['wd'],
        config['dropout'],
        config['p'],
        config['sd'])
    if not os.path.exists(dpath):
        os.makedirs(dpath)
    val_str = f'val {np.mean(val_loss):.6f} +- {np.std(val_loss):.6f}'
    test_str = f'test {np.mean(test_loss):.6f} +- {np.std(test_loss):.6f}'
    print(val_str)
    print(test_str)
    write(os.path.join(dpath, 'results.txt'), val_str)
    write(os.path.join(dpath, 'results.txt'), test_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_folds', type=int, default=7)
    parser.add_argument('--num_itrs', type=int, default=45000)
    parser.add_argument('--num_val_itrs', type=int, default=1000)
    parser.add_argument('--num_early_stop_itrs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--p', type=float, default=0)
    parser.add_argument('--sd', type=float, default=0)
    config = vars(parser.parse_args())
    main(config)