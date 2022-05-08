import datetime

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import pandas as pd

import numpy as np


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class DataLoaderH(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, horizon, window, tra_date, val_date, tes_date, normalize=2):
        self.P = window
        self.h = horizon
        self.tra_date = datetime.datetime.strptime(tra_date, "%Y-%m-%d").date()
        self.val_date = datetime.datetime.strptime(val_date, "%Y-%m-%d").date()
        self.tes_date = datetime.datetime.strptime(tes_date, "%Y-%m-%d").date()

        self.rawdat = pd.read_csv(file_name)
        p = self.rawdat.groupby("kdcode")['dt'].count() == (self.rawdat['dt'].nunique())
        p = p.reset_index()
        stock_list = p[p['dt'] == True]['kdcode'].to_list()
        self.rawdat = self.rawdat[self.rawdat.kdcode.isin(stock_list)]
        self.rawdat.sort_values(by=["kdcode", "dt"], inplace=True)

        ex_list = ['kdcode','dt',f't{horizon}_close_return_rate']
        features = list(set(self.rawdat.columns.to_list())-set(ex_list))
        self.rawdat['dt'] = self.rawdat['dt'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date())
        self.rawdat.dropna(inplace=True)
        self.rawdat.sort_values('dt', inplace=True)
        self.rawdat.reset_index(inplace=True, drop=True)
        self.sd = self.rawdat[['kdcode', 'dt']]
        for f in features:
            self.rawdat[f] = self.filter_extreme_3sigma(self.rawdat[f])
            self.rawdat[f] = self.standardize_zscore(self.rawdat[f])
            self.rawdat[f] = self.min_max_Normalization(self.rawdat[f])
        train = self.rawdat[self.rawdat.dt == self.val_date].index[0]
        valid = self.rawdat[self.rawdat.dt == self.tes_date].index[0]
        self.rawdat = self.rawdat[
            features+[f"t{horizon}_close_return_rate"]]
        self.n = self.rawdat.shape[0]
        self.m = self.rawdat.shape[1]
        self.dat = np.zeros((self.n, self.m))
        self.label_column = self.rawdat[f't{horizon}_close_return_rate']
        self.rawdat = self.rawdat.values
        self.normalize = 0
        self.scale = np.ones(self.m)
        self.bias = np.zeros(self.m)
        self._normalized(normalize)
        self._split(train, valid)
        self.sd = self.sd.iloc[range(valid, self.n), :]


        self.scale = torch.from_numpy(self.scale).float()
        self.bias = torch.from_numpy(self.bias).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.h, self.m)

        self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)
        self.bias = self.bias.cuda()
        self.bias = Variable(self.bias)

        tmp = tmp[:, -1, :].squeeze()
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def filter_extreme_3sigma(self, series, n=3):  # 3 sigma
        mean = series.mean()
        std = series.std()
        max_range = mean + n * std
        min_range = mean - n * std
        return np.clip(series, min_range, max_range)

    def standardize_zscore(self, series):
        std = series.std()
        mean = series.mean()
        # 如果标准差为0的series，全部返回0
        if std == 0:
            return series - mean
        else:
            return (series - mean) / std

    def min_max_Normalization(self, series):
        max_ = series.max()
        min_ = series.min()
        return (series - min_) / (max_ - min_)

    def _normalized(self, normalize):

        if (normalize == 0):
            self.dat = self.rawdat

        if (normalize == 1):
            # normalized by the maximum value of entire matrix.
            self.dat = self.rawdat / np.max(self.rawdat)

        if (normalize == 2):
            # normlized by the maximum value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

        if (normalize == 3):
            # normlized by the mean/std value of each row (sensor).
            for i in range(self.m):
                self.scale[i] = np.std(self.rawdat[:, i])  # std
                self.bias[i] = np.mean(self.rawdat[:, i])
                self.dat[:, i] = (self.rawdat[:, i] - self.bias[i]) / self.scale[i]
            # self.dat[:, -1] = self.rawdat[:,-1]

    def _split(self, train, valid):
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _split_stock(self, train, valid):

        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify_stock(train_set, self.h)
        self.valid = self._batchify_stock(valid_set, self.h)
        self.test = self._batchify_stock(test_set, self.h)

    def _batchify_stock(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.h, 1))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            # Y[i, :, :] = torch.from_numpy(self.dat[idx_set[i] - self.h:idx_set[i], :])
            Y[i, :, :] = torch.from_numpy(self.label_column[end:(idx_set[i] + 1)].values.reshape(1,1))

        return [X, Y]

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.h, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            # Y[i, :, :] = torch.from_numpy(self.dat[idx_set[i] - self.h:idx_set[i], :])
            Y[i, :, :] = torch.from_numpy(self.dat[end:(idx_set[i] + 1), :])

        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            X = X.cuda()
            Y = Y.cuda()
            yield Variable(X), Variable(Y)
            start_idx += batch_size
