import argparse

from train import MLPTrainer
from utils import *

def main(config):
    set_seed(config['seed'])
    dpath = 'results/mlp_{}_{}_{}_{}_{}/{}'.format(
        config['num_itrs'],
        config['batch_size'],
        config['lr'],
        config['wd'],
        config['dropout'],
        config['seed'])
    os.makedirs(dpath)
    val_folds, test_folds = get_folds(config['num_folds'])
    val_loss, test_loss = [], []
    for val_fold, test_fold in zip(val_folds, test_folds):
        config['val_fold'] = val_fold
        config['test_fold'] = test_fold
        trainer = MLPTrainer(config)
        val_loss.append(trainer.train())
        test_loss.append(trainer.test())
    results = f'{np.mean(val_loss):.6f},{np.mean(test_loss):.6f}'
    write(os.path.join(dpath, 'results.txt'), results)
    print(results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_folds', type=int, default=7)
    parser.add_argument('--num_itrs', type=int, default=50000)
    parser.add_argument('--num_val_itrs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    config = vars(parser.parse_args())
    config['num_early_stop_itrs'] = int(config['num_itrs'] / 5)
    main(config)