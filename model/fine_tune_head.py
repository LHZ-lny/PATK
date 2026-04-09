import torch.nn as nn


class Classify(nn.Module):
    def __init__(self, opt, cls_num, feat_dim=512):
        super(Classify, self).__init__()

        self.fc1 = nn.Linear(opt.d_model * opt.sampling_num * 2, 256)
        self.fc2 = nn.Linear(256, cls_num)

    def forward(self, feature):
        x = self.fc1(feature)
        x = nn.ReLU()(x)
        return self.fc2(x)


class Flatten_Head(nn.Module):
    def __init__(self, opt, d_model, pred_len, head_dropout=0.1):
        super().__init__()
        input_channels = d_model * (opt.sampling_num + 2)
        self.linear1 = nn.Linear(input_channels, d_model)
        self.linear2 = nn.Linear(d_model, pred_len * 2)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        # x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


class Predictor(nn.Module):
    def __init__(self, opt, num_layers=4, kernel_size=3, hidden_size=512):
        super(Predictor, self).__init__()
        self.opt = opt
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear((self.opt.sampling_num * 2) * self.opt.d_model, self.opt.pred_len)
        self.dropout = nn.Dropout(self.opt.dropout_rate)

    def forward(self, input_data):
        x = self.flatten(input_data)
        x = self.linear(x)
        x = self.dropout(x)
        return x