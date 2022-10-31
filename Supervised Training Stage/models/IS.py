import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList


class DeepConvLSTM(nn.Module):
    def __init__(self, seq_length, hidden_size, filter_size, sensor_num, device, label_len, conv_layer_num=5,
                 lstm_layer_num=1, feature_channel=64):
        super(DeepConvLSTM, self).__init__()
        self.encoder = Encoder(seq_length, hidden_size, filter_size, sensor_num, device, label_len, conv_layer_num,
                 lstm_layer_num, feature_channel)
    def forward(self, input):
        output = self.encoder(input)
        return output


class Encoder(nn.Module):
    def __init__(self, seq_length, hidden_size, filter_size, sensor_num, device, label_len, conv_layer_num=5,
                 lstm_layer_num=1, feature_channel=64):
        super(Encoder, self).__init__()
        self.ConvLSTM1 = ConvLSTM(seq_length, hidden_size, filter_size, device, feature_channel * 24, label_len,
                                  conv_layer_num,
                                  lstm_layer_num, feature_channel)
        self.linear1 = nn.Linear(hidden_size * 2, int(sensor_num / 2 * 6))
        self.IS = IS(seq_length, hidden_size, filter_size, sensor_num, device, feature_channel * 12, conv_layer_num=4,
                     lstm_layer_num=1, feature_channel=64)

    def forward(self, input):
        # print("0",input.shape)
        ori = self.IS(input)
        # print("1",ori.shape)
        input = torch.cat([input, ori], 2)
        # print("2",input.shape)
        output = self.ConvLSTM1(input)
        # print("3",output.shape)
        joint_angle = self.linear1(output)
        # print("4",joint_angle.shape)
        return joint_angle, ori


class IS(nn.Module):
    def __init__(self, seq_length, hidden_size, filter_size, sensor_num, device, lstm_input, conv_layer_num=5,
                 lstm_layer_num=1, feature_channel=64):
        super(IS, self).__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.lstm_layer_num = lstm_layer_num
        self.filter_size = filter_size
        self.conv_layer_num = conv_layer_num
        self.device = device
        self.conv1 = nn.Conv2d(1, feature_channel, filter_size, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_channel)
        conv_block = ConvBlock(feature_channel, filter_size)
        if (conv_layer_num - 1):
            self.convs = _get_clones(conv_block, conv_layer_num - 1)
        self.lstm = nn.LSTM(lstm_input, hidden_size, lstm_layer_num, batch_first=True,
                            bidirectional=True)
        self.bn2 = nn.BatchNorm1d(seq_length)
        self.ln = nn.LayerNorm(hidden_size * 2)
        self.linear2 = nn.Linear(hidden_size * 2, int(sensor_num * 6))

    def forward(self, input):
        batch_size = input.shape[0]
        x = input.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        if self.conv_layer_num > 1:
            for layer in self.convs:
                x = layer(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, self.seq_length, -1)

        h0 = torch.zeros(self.lstm_layer_num * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.lstm_layer_num * 2, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.lstm(x, (h0, c0))
        ori = self.bn2(x)
        ori = self.linear2(ori)
        return ori


class ConvLSTM(nn.Module):
    def __init__(self, seq_length, hidden_size, filter_size, device, lstm_input, label_len, conv_layer_num=5,
                 lstm_layer_num=1, feature_channel=64):
        super(ConvLSTM, self).__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.lstm_layer_num = lstm_layer_num
        self.filter_size = filter_size
        self.conv_layer_num = conv_layer_num
        self.feature_channel = feature_channel
        self.device = device

        self.conv1 = nn.Conv2d(1, feature_channel, filter_size, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_channel)
        conv_block = ConvBlock(feature_channel, filter_size)
        if (conv_layer_num - 1):
            self.convs = _get_clones(conv_block, conv_layer_num - 1)
        self.lstm = nn.LSTM(lstm_input, hidden_size, lstm_layer_num, batch_first=True,
                            bidirectional=True)
        self.linear3 = nn.Linear(seq_length, label_len)
        self.ln = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        if self.conv_layer_num > 1:
            for layer in self.convs:
                x = layer(x)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, self.seq_length, -1)

        h0 = torch.zeros(self.lstm_layer_num * 2, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.lstm_layer_num * 2, x.size(0), self.hidden_size).to(self.device)
        x, _ = self.lstm(x, (h0, c0))
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        x = self.linear3(x)
        x = x.permute(0, 2, 1)
        return x


class ConvBlock(nn.Module):
    def __init__(self, feature_channel, filter_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(feature_channel, feature_channel, filter_size, padding=1)
        self.bn = nn.BatchNorm2d(feature_channel)

    def forward(self, x):
        x = self.bn(F.relu(self.conv(x)))
        return x


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])
