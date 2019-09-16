import mlp
import numpy as np
import torch
import torch.nn.functional as F

from copy import deepcopy
from data_reader import BatchGenerator
from sklearn.metrics import r2_score
from torch.optim import Adam
from utils import *

class MLPTrainer:
    def __init__(self, config):
        self.config = config
        self.set_data()
        self.net = mlp.MLP(config).cuda()
        self.optimizer = Adam(self.net.parameters(), lr=config['lr'], weight_decay=config['wd'])
        self.best_val_score = -np.inf
        self.best_val_itr = None

    def set_data(self):
        data_dpath = os.path.join(os.environ['DATA_PATH'], 'xtx')
        train_fpaths = []
        for i in range(self.config['num_folds']):
            if i == self.config['val_fold']:
                val_fpath = os.path.join(data_dpath, '{}.pkl'.format(i))
            elif i == self.config['test_fold']:
                test_fpath = os.path.join(data_dpath, '{}.pkl'.format(i))
            else:
                train_fpaths.append(os.path.join(data_dpath, '{}.pkl'.format(i)))
        x_train = []
        for train_path in train_fpaths:
            x_elem, _ = load_file(train_path)
            x_train.append(x_elem)
        x_train = np.concatenate(x_train, axis=0)
        x_mean = np.nanmean(x_train, axis=0)
        x_sd = np.nanstd(x_train, axis=0)
        self.data_train = []
        for train_fpath in train_fpaths:
            self.data_train.append(BatchGenerator(self.config, train_fpath, x_mean, x_sd))
        self.data_val = BatchGenerator(self.config, val_fpath, x_mean, x_sd)
        self.data_test = BatchGenerator(self.config, test_fpath, x_mean, x_sd)

    def train(self):
        self.net.train()
        for itr in range(self.config['num_itrs']):
            self.itr = itr
            loss = 0
            for train_fold in range(len(self.data_train)):
                x, y = self.data_train[train_fold].get_batch()
                if np.random.random() < self.config['p']:
                    sd = np.random.uniform(0, self.config['sd'])
                    noise = torch.zeros_like(x).data.normal_(0, sd)
                    x = x + noise
                loss += F.mse_loss(self.net(x), y, reduction='none')
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            if itr > 0 and itr % self.config['num_val_itrs'] == 0:
                is_early_stop = self.val()
                if is_early_stop:
                    break
        return self.best_val_score

    def val(self):
        score = self.eval(self.data_val)
        if score > self.best_val_score:
            self.best_val_score = score
            self.best_val_itr = self.itr
            self.best_val_params = deepcopy(self.net.state_dict())
            return False
        else:
            if self.itr - self.best_val_itr >= self.config['num_early_stop_itrs']:
                return True

    def test(self):
        self.net.load_state_dict(self.best_val_params)
        return self.eval(self.data_test)

    def eval(self, data):
        self.net.eval()
        count = 0
        y_hat = []
        y = []
        num_examples = len(data.x)
        while count < num_examples:
            x_batch, y_batch = data.get_batch()
            y_hat.append(self.net(x_batch).data.cpu().numpy())
            y.append(y_batch.data.cpu().numpy())
            count += len(x_batch)
        y_hat, y = np.concatenate(y_hat), np.concatenate(y)
        return r2_score(y, y_hat)

class MLDGTrainer(MLPTrainer):
    def __init__(self, config):
        MLPTrainer.__init__(self, config)

    def train(self):
        self.net.train()
        for itr in range(self.config['num_itrs']):
            self.itr = itr
            val_fold = np.random.choice(a=np.arange(0, len(self.data_train)), size=1)[0]
            data_val = self.data_train[val_fold]
            meta_train_loss = 0
            for index in range(len(self.data_train)):
                if index == val_fold:
                    continue
                x, y = self.data_train[index].get_batch()
                meta_train_loss += F.mse_loss(self.net(x), y)
            x, y = data_val.get_batch()
            y_hat = self.net(
                x=x,
                meta_loss=meta_train_loss,
                meta_step_size=self.config['meta_lr'],
                stop_gradient=self.config['stop_gradient'])
            meta_loss = F.mse_loss(y_hat, y)
            total_loss = meta_train_loss + meta_loss * self.config['meta_loss_mult']
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            if itr > 0 and itr % self.config['num_val_itrs'] == 0:
                is_early_stop = self.val()
                if is_early_stop:
                    break
        return self.best_val_score