import sys
import torch
import random
from torch.nn.parallel import DataParallel

import numpy as np
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import matplotlib.pyplot as plt
import contrastive_loss_V2
import data_load.dataset_loader as data_pre

from sklearn.manifold import TSNE
from tqdm import tqdm
from model.model_NIPS_cls import Time_transformer, RoV_time, SEI_freq
from model.fine_tune_head import Classify, Predictor
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

cuda = torch.cuda.is_available()
device = torch.device('cuda')


class Pre_train:
    def __init__(self, opt):
        self.opt = opt

    def data_loading(self):
        print("Start %s_data loading..." % self.opt.scene)
        train_loader = Data.DataLoader(data_pre.Pre_train_dataset(self.opt, mode='train'), self.opt.batch_size,
                                       shuffle=True)
        print('Data loading completed!')
        return train_loader

    def model_setup(self):
        if self.opt.dataset == 'Cls':
            if self.opt.model == 'Transformer':
                model = Time_transformer(self.opt)
            elif self.opt.model == 'Mamba':
                model = Mamba(self.opt)
            else:
                raise ValueError("Do not exist the backbone!")
        else:
            raise ValueError('Do not exist this task !')
        SEI_encoder = SEI_freq(self.opt)
        RoV_encoder = RoV_time(self.opt)

        criterion_supcl = contrastive_loss_V2.SupConLoss(self.opt, temperature=self.opt.temp)
        criterion_pred = nn.L1Loss()
        device_ids = list(range(torch.cuda.device_count()))
        if len(device_ids) > 1:
            model = DataParallel(model, device_ids=device_ids).to(device_ids[0])
            SEI_encoder = DataParallel(SEI_encoder, device_ids=device_ids).to(device_ids[0])
            RoV_encoder = DataParallel(RoV_encoder, device_ids=device_ids).to(device_ids[0])
            criterion_supcl = criterion_supcl.cuda()
            criterion_pred = criterion_pred.cuda()
        else:
            model = model.cuda()
            SEI_encoder = SEI_encoder.cuda()
            RoV_encoder = RoV_encoder.cuda()
            criterion_supcl = criterion_supcl.cuda()
            criterion_pred = criterion_pred.cuda()
        return model, SEI_encoder, RoV_encoder, criterion_supcl, criterion_pred

    def optimizer_setup(self, model, SEI_encoder, RoV_encoder):
        if self.opt.model == 'Transformer':
            optimizer = optim.Adadelta(
                list(model.parameters()) + list(RoV_encoder.parameters()) + list(SEI_encoder.parameters()),
                lr=self.opt.learning_rate)
        elif self.opt.model == 'Mamba':
            optimizer = optim.AdamW(
                list(model.parameters()) + list(RoV_encoder.parameters()) + list(SEI_encoder.parameters()),
                lr=self.opt.learning_rate)
        return optimizer

    def pre_train(self):
        train_loader = self.data_loading()
        model, SEI_encoder, RoV_encoder, criterion_supcl, criterion_pred = self.model_setup()
        optimizer = self.optimizer_setup(model, SEI_encoder, RoV_encoder)

        # fine_tuner = Cls_fine_tune(self.opt)

        writer: SummaryWriter = SummaryWriter(log_dir='V2_logs_%s' % self.opt.pre_dataset_name)

        for epoch in range(self.opt.EPOCH):
            total_loss = self.Cls_pre_train_one_epoch(train_loader, model, SEI_encoder, RoV_encoder,
                                                      criterion_supcl, optimizer, epoch)
            writer.add_scalar('Total_loss', total_loss, epoch + 1)
            # writer.add_scalar('Time_Cl_loss', time_cl_loss, epoch + 1)
            # writer.add_scalar('Freq_Cl_loss', freq_cl_loss, epoch + 1)
            with torch.no_grad():
                if (epoch + 1) % self.opt.saved_interval == 0:
                    torch.save(model.state_dict(),
                               "./%s_pre_train_%s/%s.pth" % (self.opt.saved_models_files, self.opt.pre_dataset_name, 'model_' + str(epoch + 1)))
                    # fine_tuner.fine_tune_classification(epoch + 1, dataset_name=self.opt.tune_dataset_name)
        writer.close()
        print('Training Complete!')

    def Cls_pre_train_one_epoch(self, train_loader, model, SEI_encoder, RoV_encoder, criterion_supcl, optimizer, epoch):
        with tqdm(total=len(train_loader), leave=True, desc="epoch" + str(epoch), ncols=150, unit='it',
                  unit_scale=True) as t:

            total_loss = []

            model.train()
            SEI_encoder.train()
            RoV_encoder.train()

            for input_data, label in train_loader:

                batch_size, data_length, channels = input_data.shape
                input_data = input_data.float().to(device)

                time_data = RoV_encoder(input_data)
                freq_data = SEI_encoder(input_data)

                outoput = model(time_data, freq_data, batch_size, data_length, channels)

                train_loss = criterion_supcl(outoput)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                total_loss.append(train_loss.item())
                # time_cl_loss.append(Time_CL_Loss.item())
                # freq_cl_loss.append(Freq_CL_Loss.item())
                # series_cl_loss.append(0)

                t.update()
                t.set_postfix(loss=['total loss:', np.round(np.mean(total_loss), 2)], lr=self.opt.learning_rate)

            total_loss = torch.tensor(total_loss).mean()
            # time_cl_loss = torch.tensor(time_cl_loss).mean()
            # freq_cl_loss = torch.tensor(freq_cl_loss).mean()

        return total_loss


class Cls_fine_tune:
    def __init__(self, opt):
        self.opt = opt

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def fine_tune_classification(self, model_num, dataset_name):

        model = Time_transformer(self.opt)
        SEI_encoder = SEI_freq(self.opt)
        RoV_encoder = RoV_time(self.opt)
        device_ids = list(range(torch.cuda.device_count()))

        assert dataset_name in ['FD-A', 'FD-B', 'Gesture', 'EMG', 'Epilepsy', 'ECG']
        cls_map = {'FD-A': 3, 'FD-B': 3, 'Gesture': 8, 'EMG': 3, 'Epilepsy': 2, 'ECG': 3}
        self.cls = cls_map[dataset_name]
        Classifier = Classify(self.opt, self.cls, feat_dim=self.opt.d_model)
        criterion_cls = nn.CrossEntropyLoss()
        optimizer = optim.Adam(list(Classifier.parameters()) + list(RoV_encoder.parameters()) + list(SEI_encoder.parameters()), lr=0.001)
        if len(device_ids) > 1:
            model = DataParallel(model, device_ids=device_ids).to(device_ids[0])
            # SEI_encoder = DataParallel(SEI_encoder, device_ids=device_ids).to(device_ids[0])
            RoV_encoder = DataParallel(RoV_encoder, device_ids=device_ids).to(device_ids[0])
            Classifier = DataParallel(Classifier, device_ids=device_ids).to(device_ids[0])
            criterion_cls = criterion_cls.cuda()
        else:
            model = model.cuda()
            SEI_encoder = SEI_encoder.cuda()
            RoV_encoder = RoV_encoder.cuda()
            Classifier = Classifier.cuda()
            criterion_cls = criterion_cls.cuda()

        model.load_state_dict(
            torch.load("./%s_pre_train_%s/%s.pth" % (self.opt.saved_models_files, self.opt.pre_dataset_name, 'model_' + str(model_num))))

        train_loader = Data.DataLoader(
            data_pre.Fine_tune_cla_dataset(self.opt, mode='train', dataset_name=dataset_name),
            4, shuffle=True)

        test_loader = Data.DataLoader(
            data_pre.Fine_tune_cla_dataset(self.opt, mode='test', dataset_name=dataset_name),
            4, shuffle=True)

        for epoch in range(self.opt.fine_tune_epochs):
            step = 0
            model.eval()
            Classifier.train()
            RoV_encoder.train()
            SEI_encoder.train()
            total_loss = []
            for input_data, label in train_loader:
                label = label.to(torch.int64)

                batch_size, data_length, channels = input_data.shape
                input_data = input_data.float().to(device)

                time_data = RoV_encoder(input_data)
                freq_data = SEI_encoder(input_data)

                outoput = model(time_data, freq_data, batch_size, data_length, channels)
                pre_feat = outoput.view(outoput.size(0), -1)
                pred = Classifier(pre_feat)
                train_loss = criterion_cls(pred, label)

                train_loss.requires_grad_(True)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                step += 1
                total_loss.append(train_loss.item())
            total_loss = torch.tensor(total_loss).mean()
            sys.stdout.write("\r[Epoch %d][Training_Loss: %f]" % (epoch, total_loss))

        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for input_data, label in test_loader:
                if input_data.shape[1] != 1:
                    input_data = torch.unsqueeze(input_data[:, 0, :], dim=1).to(device)

                batch_size, data_length, channels = input_data.shape
                input_data = input_data.float().to(device)

                time_data = RoV_encoder(input_data)
                freq_data = SEI_encoder(input_data)

                outoput = model(time_data, freq_data, batch_size, data_length, channels)
                pre_feat = outoput.view(outoput.size(0), -1)
                pred = Classifier(pre_feat)

                true_labels.extend(label.detach().cpu().numpy())
                predicted_labels.extend(pred.argmax(1).detach().cpu().numpy())
            # 计算准确率
            accuracy = accuracy_score(true_labels, predicted_labels)

            # 计算精确度
            precision = precision_score(true_labels, predicted_labels, average='macro')

            # 计算召回率
            recall = recall_score(true_labels, predicted_labels, average='macro')  # 在多类别问题中使用weighted参数

            f1 = f1_score(true_labels, predicted_labels, average='macro')

            # 计算Kappa系数
            kappa = cohen_kappa_score(true_labels, predicted_labels)

            print("\r", 'dataset:', dataset_name, ':', 'Acc:', accuracy, 'Pre:',
                  precision, 'Re:', recall, 'F1:', f1, 'Kap:', kappa)

    def add_gaussian_noise(self, input_data, desired_snr):
        if torch.is_tensor(input_data):
            # 计算信号的功率
            signal_power = torch.mean(input_data.pow(2))

            # 计算所需的噪声功率
            noise_power = signal_power / (10 ** (desired_snr / 10))

            # 生成高斯噪声
            noise = torch.randn_like(input_data) * torch.sqrt(noise_power)

            # 添加噪声到信号中
            noisy_signal = input_data + noise

        elif isinstance(input_data, (list, tuple, np.ndarray)):
            signal_power = np.mean(np.square(input_data))

            # 计算所需的噪声功率
            noise_power = signal_power / (10 ** (desired_snr / 10))

            # 生成高斯噪声
            noise = np.random.normal(0, np.sqrt(noise_power), size=input_data.shape)

            # 添加噪声到信号中
            noisy_signal = input_data + noise

        else:
            raise TypeError("The input signal must be tensor, list, tuple or numpy!")

        return noisy_signal

    def zero_out_segments(self, input_data, num_segments_to_zero):
        # 确定信号的长度
        signal_length = len(input_data)

        # 生成随机序列，表示哪些部分需要置零
        zero_indices = np.random.choice(signal_length, size=num_segments_to_zero, replace=False)

        # 将选定的部分置零
        noisy_signal = input_data.copy()
        for index in zero_indices:
            noisy_signal[index] = 0.0

        return noisy_signal

    # MSE
    def mean_squared_error(self, predictions, targets):
        mse = ((predictions - targets) ** 2).mean()
        return mse

    # MAE
    def mean_absolute_error(self, predictions, targets):
        mae = torch.abs(predictions - targets).mean()
        return mae
