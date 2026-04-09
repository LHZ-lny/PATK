import csv
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data
import pandas as pd
from data_load.utils import time_features
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda')


class Pre_train_dataset(Data.Dataset):
    def __init__(self, opt, mode='train'):
        self.opt = opt

        if mode == 'train':

            train_file_name = './dataset/%s/%s/train.pt' % (opt.dataset, opt.pre_dataset_name)
            train = torch.load(train_file_name)
            self.samples = train['samples'].to(device)
            self.labels = train['labels'].to(device)

        elif mode == 'test':

                test_file_name = './dataset/%s/%s/test.pt' % (opt.dataset, opt.pre_dataset_name)
                test = torch.load(test_file_name)
                self.samples = test['samples'].to(device)
                self.labels = test['labels'].to(device)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

    def __len__(self):
        return len(self.samples)


class Fine_tune_cla_dataset(Data.Dataset):
    def __init__(self, opt, mode='train', dataset_name='FD-B'):
        self.opt = opt

        if mode == 'train':

            train_file_name = './dataset/%s/%s/train.pt' % (opt.dataset, opt.tune_dataset_name)

            train = torch.load(train_file_name)

                # train['samples'] = torch.Tensor(train['samples'])
                # train['labels'] = torch.Tensor(train['labels'])
                # torch.save(train, train_file_name)

            if isinstance(train['samples'], torch.Tensor):
                self.samples = train['samples'].to(device)
                self.labels = train['labels'].to(device)
                #
                # sample = train['samples'][0].detach().numpy()[0]
                # time = np.linspace(0, 1, len(sample))
                # plt.figure(0)
                # plt.plot(time, sample)
                # plt.show()
                # a = 0
            else:
                self.samples = torch.Tensor(train['samples']).to(device)
                self.labels = torch.Tensor(train['labels']).to(device)

        elif mode == 'test':

            test_file_name = './dataset/%s/%s/test.pt' % (opt.dataset, dataset_name)
            test = torch.load(test_file_name)
            # test['samples'] = torch.Tensor(test['samples'])
            # test['labels'] = torch.Tensor(test['labels'])
            # torch.save(test, test_file_name)
            if isinstance(test['samples'], torch.Tensor):
                self.samples = test['samples'].to(device)
                self.labels = test['labels'].to(device)
            else:
                self.samples = torch.Tensor(test['samples']).to(device)
                self.labels = torch.Tensor(test['labels']).to(device)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

    def __len__(self):
        return len(self.samples)


class Dataset_ETT_hour(Data.Dataset):
    def __init__(self, opt, mode='train', features='MS', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=1, freq='h', dataset_type='ETT'):
        # size [seq_len, label_len, pred_len]
        # info

        self.opt = opt

        if self.opt.input_len is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = self.opt.input_len
            self.label_len = self.opt.pred_len
            self.pred_len = self.opt.pred_len
        # init

        assert mode in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[mode]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        if self.opt.scene == 'pre-train':
            self.root_path = './dataset/%s/%s' % (self.opt.dataset, dataset_type)
        elif self.opt.scene == 'fine-tune':
            self.root_path = './dataset/%s/%s' % (self.opt.dataset, dataset_type)

        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            pass
            # data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            # data_stamp = data_stamp.transpose(1, 0)

        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], 1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # print(type(self.data_x))
        # print(self.data_x.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class ETT_dataset(Data.Dataset):
    def __init__(self, opt, target_col='HUFL', mode='train', transform=None, dataset_type='ETT', dataset_name='ETTh1'):
        assert mode in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[mode]

        self.opt = opt
        self.dataset_name = dataset_name
        self.file_path = './dataset/%s/%s/%s.csv' % (self.opt.dataset, dataset_type, self.dataset_name)
        self.scaler = True
        if self.opt.input_len is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = self.opt.input_len
            self.label_len = self.opt.pred_len
            self.pred_len = self.opt.pred_len

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.target_col = target_col
        self.transform = transform
        self.data = pd.read_csv(self.file_path)

        cols_data = self.data.columns[1:]
        df_data = self.data[cols_data]

        data = df_data.values

        df_stamp = self.data[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], 1).values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # print(type(self.data_x))
        # print(self.data_x.shape)

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end].astype(np.float64)
        seq_y = self.data_y[r_begin:r_end].astype(np.float64)
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x = torch.Tensor(seq_x).to(device)
        seq_y = torch.Tensor(seq_y).to(device)
        # seq_x_mark = torch.Tensor(seq_x_mark).to(device)
        # seq_y_mark = torch.Tensor(seq_y_mark).to(device)
        seq_x = torch.transpose(seq_x, 0, 1)
        seq_y = torch.transpose(seq_y, 0, 1)
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
