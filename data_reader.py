import numpy as np
import os
import pandas as pd
import torch

from utils import save_file, load_file

class BatchGenerator:
    def __init__(self, config, fpath, x_mean, x_sd):
        self.config = config
        self.x, self.y = load_file(fpath)
        self.x -= x_mean
        self.x /= x_sd
        self.x = np.nan_to_num(self.x)
        self.file_num_train = len(self.x)
        self.current_index = -1
        self.shuffle()

    def shuffle(self):
        idxs = np.random.permutation(len(self.x))
        self.x, self.y = self.x[idxs], self.y[idxs]

    def get_batch(self):
        x = []
        y = []
        for index in range(self.config['batch_size']):
            self.current_index += 1
            if self.current_index > self.file_num_train - 1:
                self.current_index %= self.file_num_train
                self.shuffle()
            x.append(self.x[self.current_index])
            y.append(self.y[self.current_index])
        x = np.stack(x)
        y = np.stack(y)
        return torch.tensor(x, dtype=torch.float).cuda(), torch.tensor(y, dtype=torch.float).cuda()

def preprocess(num_folds):
    data_dpath = os.path.join(os.environ['DATA_PATH'], 'xtx')
    data = pd.read_csv(os.path.join(data_dpath, 'data-training.csv')).values
    x, y = data[:, :-1], data[:, -1]
    del data
    rem = len(x) % num_folds
    idxs = np.arange(rem, len(x))
    domain_idxs_list = np.split(idxs, num_folds)
    for i, domain_idxs in enumerate(domain_idxs_list):
        save_file((x[domain_idxs], y[domain_idxs]), os.path.join(data_dpath, '{}.pkl'.format(i)))