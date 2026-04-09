import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from random import *
from einops import rearrange
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legweight
from torch.func import vmap

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len, seq_k = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q[:, :, 0].data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    output = pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]
    return output


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


# class MCMC(nn.Module):
#     def __init__(self, opt):
#         super(MCMC, self).__init__()
#         self.sampling_num = opt.sampling_num
#         self.time_sampling_length = opt.sampling_length
#
#     def forward(self, data):
#         device = (torch.device('cuda')
#                   if data.is_cuda
#                   else torch.device('cpu'))
#         batch_size = data.shape[0]
#
#         slice_data = torch.zeros((batch_size, self.sampling_num, self.sampling_length)).to(device)
#
#         for i in range(self.sampling_num):
#             initial = randrange(0, data.shape[2])
#             end = initial + self.sampling_length
#             if end > data.shape[2]:
#                 slice_data[:, i, :] = torch.cat([data[:, 0, initial:], torch.zeros(
#                     (batch_size, end - data.shape[2])).to(device)], dim=1)
#             else:
#                 slice_data[:, i, :] = data[:, 0, initial: end]
#
#         return slice_data


class SEI_freq(nn.Module):
    def __init__(self, opt):
        super(SEI_freq, self).__init__()
        self.opt = opt
        self.sampling_num = opt.sampling_num
        self.sampling_length = opt.sampling_length

        self.length_weights = nn.Sequential(
            nn.Linear(self.opt.sampling_length, 1),
            nn.Tanh()
        )

        self.sign_weights = nn.Sequential(
            nn.Linear(self.opt.sampling_length, 1),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.embedding = nn.Linear(self.sampling_length, self.opt.d_model)

    def forward(self, data, norm=1, keepdim=True, sigma=1E-6, mode='linear'):
        """
        data: shape [batch_size, 1, length]
        """

        data_freq = torch.fft.fftshift(data)

        batch_size = data.shape[0]
        length = data.shape[2]

        # 计算能量分布
        diffs = torch.abs(data_freq) ** 2 + sigma
        change_rates = torch.norm(diffs, p=norm, dim=-1, keepdim=keepdim)
        probabilities = diffs / change_rates
        first_rate = probabilities.mean(dim=-1, keepdim=keepdim)
        probabilities = torch.cat([first_rate, probabilities], dim=-1).squeeze(1)[:, :length - self.sampling_length]

        # 生成初始点位置
        start_indices = torch.multinomial(probabilities, self.sampling_num, replacement=False)
        start_indices, _ = torch.sort(start_indices, dim=1)

        # 生成切片索引
        base_indices = torch.arange(self.sampling_length, device=start_indices.device).expand(
            batch_size * self.sampling_num, self.sampling_length)
        indices = start_indices.view(batch_size * self.sampling_num, 1) + base_indices  # 随机patch每个数据点的index

        # 使用索引提取切片数据
        slice_data = data_freq.expand(-1, self.sampling_num, -1)  # [batch_size,sampling_num,length]
        slice_data = slice_data.reshape(batch_size * self.sampling_num, length).to(self.opt.device)
        slice_data = torch.gather(slice_data, dim=1, index=indices)
        slice_data = slice_data.view(batch_size, self.sampling_num, self.sampling_length)

        # 长度伸缩
        length_weights = 0.5 * self.length_weights(slice_data).reshape(-1, 1)
        slice_data = slice_data.view(batch_size * self.sampling_num, -1)
        elasctic_length = torch.round(
            torch.tensor(self.sampling_length).to(self.opt.device).expand(batch_size * self.sampling_num, 1) * (
                        1 + length_weights)).squeeze(-1).type(torch.int64)
        for i in range(batch_size * self.sampling_num):
            slice_data[i, :] = F.interpolate(slice_data[i, :elasctic_length[i]].unsqueeze(0).unsqueeze(0),
                                             size=self.sampling_length,
                                             mode=mode if mode != 'cubic' else 'linear',  # 1D不支持cubic，用linear替代
                                             align_corners=True if mode != 'nearest' else None
                                             ).squeeze(0).squeeze(0)
        slice_data = slice_data.view(batch_size, self.sampling_num, self.sampling_length)

        # 重要性加权操作
        slice_data = self.sign_weights(slice_data) * slice_data
        freq_embedding = self.embedding(slice_data)
        return freq_embedding  # [batch_size,sampling_num,sampling_lengths]


class Series_encode(nn.Module):
    def __init__(self, input_size, output_size):
        super(Series_encode, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear1(x)
        return out


class RoV_time(nn.Module):
    def __init__(self, opt):
        super(RoV_time, self).__init__()

        self.opt = opt
        self.sampling_num = opt.sampling_num
        self.sampling_length = opt.sampling_length

        self.length_weights = nn.Sequential(
            nn.Linear(self.opt.sampling_length, 1),
            nn.Tanh()
        )

        self.sign_weights = nn.Sequential(
            nn.Linear(self.opt.sampling_length, 1),
            nn.ReLU(),
            nn.Sigmoid()
        )
        self.embedding = nn.Linear(self.sampling_length, self.opt.d_model)

    def forward(self, data, norm=1, keepdim=True, sigma=1E-6, mode='linear', align_corners=True):
        """
        data: shape [batch_size, 1, length]
        """

        batch_size = data.shape[0]
        length = data.shape[2]

        # 计算变化率分布
        diffs = torch.abs(torch.diff(data, dim=-1)) + sigma
        change_rates = torch.norm(diffs, p=norm, dim=-1, keepdim=keepdim)
        probabilities = diffs / change_rates
        first_rate = probabilities.mean(dim=-1, keepdim=keepdim)
        probabilities = torch.cat([first_rate, probabilities], dim=-1).squeeze(1)[:, :length - self.sampling_length]

        # -------------------------- 2. 优化初始点生成 --------------------------
        start_indices = torch.multinomial(probabilities, self.sampling_num, replacement=False)  # [B, sampling_num]
        start_indices, _ = torch.sort(start_indices, dim=1)

        # 生成索引（简化广播，减少内存开销）
        base_indices = torch.arange(self.sampling_length, device=data.device)  # [sampling_length]
        indices = start_indices.view(-1, 1) + base_indices  # [B*sampling_num, sampling_length]

        # -------------------------- 3. 优化切片提取 --------------------------
        slice_data = data.expand(-1, self.sampling_num, -1)  # [B, S, L]，S=sampling_num
        slice_data = slice_data.reshape(-1, length)  # [B*S, L]
        slice_data = torch.gather(slice_data, dim=1, index=indices)  # [B*S, sampling_length]

        # -------------------------- 4. 优化长度伸缩（适配PyTorch 1.9，核心修改） --------------------------
        # 计算伸缩权重 & 目标长度（clamp防止索引越界）
        length_weights = 0.5 * self.length_weights(slice_data.unsqueeze(1)).reshape(-1, 1)  # [B*S, 1]
        elasctic_length = torch.round(self.sampling_length * (1 + length_weights)).to(torch.int64)
        elasctic_length = elasctic_length.clamp(min=1, max=self.sampling_length).squeeze(-1)  # [B*S]

        # 优化循环：1. 提前将所有数据移到GPU并保持在GPU 2. 减少循环内张量创建 3. 预分配结果张量
        total_num = batch_size * self.sampling_num
        # 预分配结果张量（避免循环内反复拼接，减少内存碎片）
        slice_data_interp = torch.empty_like(slice_data, device=slice_data.device)

        # 优化循环：将循环内的重复操作提前提取，减少计算量
        target_size = self.sampling_length

        # 仅在GPU上执行循环（减少GPU-CPU交互）
        with torch.no_grad():  # 若length_weights无梯度，可加no_grad进一步提速；有梯度则注释
            for i in range(total_num):
                # 提取当前样本的有效长度（避免循环内重复索引）
                curr_len = elasctic_length[i]
                # 切片+插值（简化张量操作，减少squeeze/unsqueeze次数）
                x_valid = slice_data[i, :curr_len].view(1, 1, -1)  # 直接reshape，替代两次unsqueeze
                x_interp = F.interpolate(
                    x_valid,
                    size=target_size,
                    mode=mode,
                    align_corners=align_corners
                )
                slice_data_interp[i] = x_interp.view(-1)  # 直接view，替代两次squeeze

        # 恢复批量维度
        slice_data = slice_data_interp.view(batch_size, self.sampling_num, self.sampling_length)

        # -------------------------- 5. 重要性加权与嵌入 --------------------------
        slice_data = self.sign_weights(slice_data) * slice_data
        time_embedding = self.embedding(slice_data)
        return time_embedding

        # # 生成初始点位置
        # start_indices = torch.multinomial(probabilities, self.sampling_num, replacement=False)
        # start_indices, _ = torch.sort(start_indices, dim=1)
        #
        # # 生成切片索引
        # base_indices = torch.arange(self.sampling_length, device=start_indices.device).expand(batch_size * self.sampling_num, self.sampling_length)
        # indices = start_indices.view(batch_size * self.sampling_num, 1) + base_indices  # 随机patch每个数据点的index
        #
        # # 使用索引提取切片数据
        # slice_data = data.expand(-1, self.sampling_num, -1) # [batch_size,sampling_num,length]
        # slice_data = slice_data.reshape(batch_size * self.sampling_num, length).to(self.opt.device)
        # slice_data =torch.gather(slice_data, dim=1, index=indices)
        # slice_data = slice_data.view(batch_size, self.sampling_num, self.sampling_length)
        #
        # # 长度伸缩
        # length_weights = 0.5 * self.length_weights(slice_data).reshape(-1, 1)
        # slice_data = slice_data.view(batch_size * self.sampling_num, -1)
        # elasctic_length = torch.round(torch.tensor(self.sampling_length).to(self.opt.device).expand(batch_size * self.sampling_num, 1) * (1 + length_weights)).squeeze(-1).type(torch.int64)
        # for i in range(batch_size * self.sampling_num):
        #     slice_data[i, :] = F.interpolate(slice_data[i, :elasctic_length[i]].unsqueeze(0).unsqueeze(0),
        #                                         size=self.sampling_length,
        #                                         mode=mode if mode != 'cubic' else 'linear',  # 1D不支持cubic，用linear替代
        #                                         align_corners=True if mode != 'nearest' else None
        #                                         ).squeeze(0).squeeze(0)
        #
        # slice_data = slice_data.view(batch_size, self.sampling_num, self.sampling_length)
        #
        # # 重要性加权操作
        # slice_data = self.sign_weights(slice_data) * slice_data
        # time_embedding = self.embedding(slice_data)
        # return time_embedding  # [batch_size,sampling_num,sampling_lengths]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.L1 = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, Q, K, V, attn_mask, n_heads, d_k, d_v):

        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask, d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            n_heads * d_v)  # context: [batch_size, seq_len, n_heads, d_v]
        output = self.L1(context)
        return output  # output: [batch_size, seq_len, d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.fc1(inputs)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        output = self.fc2(x)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout_rate)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.drop_out = nn.Dropout(dropout_rate)

    def forward(self, enc_inputs, enc_self_attn_mask, n_heads, d_k, d_v):
        attention_outoput = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask, n_heads, d_k, d_v)  # enc_inputs to same Q,K,V
        x = enc_inputs + self.drop_out(attention_outoput)
        x = self.layer_norm1(x)

        feed_forward_output = self.pos_ffn(x)  # enc_outputs: [batch_size, seq_len, d_model]
        x = x + self.drop_out(feed_forward_output)
        enc_outputs = self.layer_norm2(x)
        return enc_outputs


class Time_transformer(nn.Module):
    def __init__(self, opt, head='mlp', feat_dim=128, channels=7):
        super(Time_transformer, self).__init__()
        self.opt = opt
        self.embedding = DataEmbedding(opt.d_model, opt.d_model, opt.dropout_rate)
        self.layers = nn.ModuleList(
            [EncoderLayer(opt.d_model, opt.d_k, opt.d_v, opt.n_heads, opt.d_ff, opt.dropout_rate) for _ in
             range(opt.n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(opt.d_model, opt.d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )

        self.time_dim_in = self.opt.d_model
        self.freq_dim_in = self.opt.d_model

        if head == 'linear':
            self.head = nn.Linear(opt.input_len, feat_dim)
        elif head == 'mlp':
            self.head_time_token = nn.Sequential(
                nn.Linear(self.time_dim_in, self.time_dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(self.time_dim_in, feat_dim)
            )

            self.head_freq_token = nn.Sequential(
                nn.Linear(self.freq_dim_in, self.freq_dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(self.freq_dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, time_data, freq_data, batch_size, data_length, channels):

        time = self.embedding(time_data)
        freq = self.embedding(freq_data)


        time_batch, time_token, time_len = time.shape
        freq_batch, freq_token, freq_len = freq.shape

        output = torch.cat([time, freq], dim=1)
        # output = time
        enc_self_attn_mask = get_attn_pad_mask(output, output)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask, self.opt.n_heads, self.opt.d_k, self.opt.d_v)
        output = self.fc(output)

        # time_token_output = output[:, :time_token, :]
        # freq_token_output = output[:, time_token:, :]

        # time_feat_tokens = F.normalize(self.head_time_token(time_token_output), dim=1)
        # freq_feat_tokens = F.normalize(self.head_freq_token(freq_token_output), dim=1)

        return output

