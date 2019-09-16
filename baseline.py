import os

from argparse import ArgumentParser
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from utils import *

def get_data(num_folds, val_fold, test_fold):
    data_dpath = os.path.join(os.environ['DATA_PATH'], 'xtx')
    x_train, y_train = [], []
    for i in range(num_folds):
        if i == val_fold:
            x_val, y_val = load_file(os.path.join(data_dpath, '{}.pkl'.format(i)))
        elif i == test_fold:
            x_test, y_test = load_file(os.path.join(data_dpath, '{}.pkl'.format(i)))
        else:
            x_train_elem, y_train_elem = load_file(os.path.join(data_dpath, '{}.pkl'.format(i)))
            x_train.append(x_train_elem)
            y_train.append(y_train_elem)
    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train)
    x_mean, x_sd = np.nanmean(x_train, axis=0), np.nanstd(x_train, axis=0)
    x_train -= x_mean
    x_train /= x_sd
    x_val -= x_mean
    x_val /= x_sd
    x_test -= x_mean
    x_test /= x_sd
    x_train = np.nan_to_num(x_train)
    x_val = np.nan_to_num(x_val)
    x_test = np.nan_to_num(x_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

def main(args):
    val_folds = np.arange(args.num_folds)
    test_folds = np.roll(val_folds, 1)
    val_losses, test_losses = [], []
    for val_fold, test_fold in zip(val_folds, test_folds):
        x_train, y_train, x_val, y_val, x_test, y_test = get_data(args.num_folds, val_fold, test_fold)
        model = Lasso(alpha=args.wd).fit(x_train, y_train)
        val_loss = r2_score(y_val, model.predict(x_val))
        test_loss = r2_score(y_test, model.predict(x_test))
        val_losses.append(val_loss)
        test_losses.append(test_loss)
        print('val {:.6f}, eval {:.6f}'.format(val_loss, test_loss))
    print(f'val {np.mean(val_losses):.6f} +- {np.std(val_losses):.6f}')
    print(f'eval {np.mean(test_losses):.6f} +- {np.std(test_losses):.6f}')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--num_folds', type=int, default=7)
    parser.add_argument('--wd', type=float, default=5e-4)
    args = parser.parse_args()
    main(args)