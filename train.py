import mlp
import torch.nn.functional as F

from data_reader import BatchGenerator
from sklearn.metrics import r2_score
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import *

class MLPTrainer:
    def __init__(self, config):
        self.config = config
        self.set_data()
        self.log_fpath = os.path.join(self.config['dpath'], 'log_{}.csv'.format(self.config['val_fold']))
        self.net = mlp.MLP(config).cuda()
        self.optimizer = Adam(self.net.parameters(), lr=config['lr'], weight_decay=config['wd'])
        self.scheduler = CosineAnnealingLR(self.optimizer, config['num_itrs'])
        self.scores = []

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
        for itr in range(1, self.config['num_itrs'] + 1):
            self.itr = itr
            loss = 0
            for fold in self.data_train:
                x, y = fold.get_batch()
                loss += F.mse_loss(self.net(x), y)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.scheduler.step(itr)
            if itr > 0 and itr % self.config['num_val_itrs'] == 0:
                self.val()
        return self.scores[-1][1]

    def val(self):
        score = self.eval(self.data_val)
        self.scores.append([self.itr, score])
        self.log()

    def test(self):
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

    def log(self):
        pd.DataFrame(np.stack(self.scores)).to_csv(self.log_fpath, header=False, index=False)

class MLDGTrainer(MLPTrainer):
    def __init__(self, config):
        MLPTrainer.__init__(self, config)

    def train(self):
        self.net.train()
        for itr in range(self.config['num_itrs']):
            self.itr = itr
            support_fold, query_fold = np.random.choice(len(self.data_train), 2, replace=False)
            data_support, data_query = self.data_train[support_fold], self.data_train[query_fold]
            x_support, y_support = data_support.get_batch()
            support_loss = F.mse_loss(self.net(x_support), y_support)
            x_query, y_query = data_query.get_batch()
            y_hat = self.net(
                x=x_query,
                meta_loss=support_loss,
                meta_step_size=self.config['inner_lr'],
                stop_gradient=self.config['stop_gradient'])
            query_loss = F.mse_loss(y_hat, y_query)
            loss = support_loss + self.config['query_loss_mult'] * query_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(itr)
            if itr > 0 and itr % self.config['num_val_itrs'] == 0:
                self.val()
        return self.scores[-1][1]