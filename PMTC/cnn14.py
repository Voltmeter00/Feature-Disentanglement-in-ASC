#-*- coding = utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
import pickle
from torch.utils.data import DataLoader
import numpy as np

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)   #均匀分布

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):


        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        x = self.dropout(x)
        return x


class Cnn14(nn.Module):
    def __init__(self,dim):

        super(Cnn14, self).__init__()

        self.bn0 = nn.BatchNorm2d(256)
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, dim, bias=True)
        self.b = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(0.5)
        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
    def forward(self, input_feature):
        """
        Input: (batch_size, data_length)"""

        x = input_feature.unsqueeze(1)  # (batch_size, 1, time_steps, freq_bins)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = self.spec_augmenter(x)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')

        # x = F.dropout(x, p=0.2)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')

        # x = F.dropout(x, p=0.2)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')

        # x = F.dropout(x, p=0.2)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')

        # x = F.dropout(x, p=0.2)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')

        # x = F.dropout(x, p=0.2)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')

        # x = F.dropout(x, p=0.2)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = self.dropout(x)
        x = F.relu_(self.fc1(x))
        x = self.b(x)


        return x


class AudioNet(nn.Module):
    def __init__(self,dim):

        super(AudioNet, self).__init__()
        self.cnn14 = Cnn14(dim)
        # state = torch.load("/home/tyz/TAU/Cnn14.pth")["model"]
        # model_dict = self.cnn14.state_dict()
        # pretrained_dict = {}
        # for k, v in state.items():
        #     if k in model_dict:
        #         if v.size() == model_dict[k].size():
        #             pretrained_dict[k] = v
        # print(pretrained_dict.keys())
        # model_dict.update(pretrained_dict)
        # self.cnn14.load_state_dict(model_dict)

    def forward(self,x):
        x = self.cnn14(x)
        return x