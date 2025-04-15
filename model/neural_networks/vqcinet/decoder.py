import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

"""
References:

Lu, Zhilin, Jintao Wang, and Jian Song. "Multi-resolution CSI feedback with deep learning in massive MIMO system." ICC 2020-2020 IEEE International Conference on Communications (ICC). IEEE, 2020.

https://github.com/Kylin9511/CRNet/blob/master/main.py
"""


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        super(ConvBN, self).__init__(OrderedDict([
            ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                               padding=padding, groups=groups, bias=False)),
            ('bn', nn.BatchNorm2d(out_planes))
        ]))

class CRBlock(nn.Module):
    def __init__(self, channel_size, original_channel_size):
        super(CRBlock, self).__init__()
        self.path1 = nn.Sequential(OrderedDict([
            ('conv3x3', ConvBN(original_channel_size, channel_size, 3)),
            ('relu1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1x9', ConvBN(channel_size, channel_size, [1, 9])),
            ('relu2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv9x1', ConvBN(channel_size, channel_size, [9, 1])),
        ]))
        self.path2 = nn.Sequential(OrderedDict([
            ('conv1x5', ConvBN(original_channel_size, channel_size, [1, 5])),
            ('relu', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv5x1', ConvBN(channel_size, channel_size, [5, 1])),
        ]))
        self.conv1x1 = ConvBN(channel_size * 2, original_channel_size, 1)
        self.identity = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x):
        identity = self.identity(x)

        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.relu(out)
        out = self.conv1x1(out)
        out = self.relu(out + identity)
        return out


class Decoder(nn.Module):
    """
    """

    def __init__(self, img_shape):
        super(Decoder, self).__init__()

        self.img_shape = img_shape
        self.n_channels = self.img_shape[0]
        self.img_size = self.img_shape[0] * self.img_shape[1] * self.img_shape[2]
        #self.codeword_length = config["n_codeword_floats"]
        #reduction = self.img_size // self.codeword_length

        feature_decoder_upsampling_layers = nn.Sequential(OrderedDict([
            ('conv', nn.ConvTranspose2d(in_channels=12, out_channels=6, kernel_size=(2, 2), stride=(2, 2), bias=False)),
            ('bn', nn.BatchNorm2d(6)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv2', nn.ConvTranspose2d(in_channels=6, out_channels=2, kernel_size=(2, 2), stride=(2, 2), bias=False)),
            ('bn2', nn.BatchNorm2d(2)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))

        self.feature_decoder_upsampling_layers = nn.Sequential(feature_decoder_upsampling_layers)


        feature_decoder = OrderedDict([
            ("conv5x5_bn", ConvBN(self.n_channels, self.n_channels, 5)),
            ("relu", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("CRBlock1", CRBlock(channel_size=9, original_channel_size=self.n_channels)),
            ("CRBlock2", CRBlock(channel_size=7, original_channel_size=self.n_channels)),
            ("CRBlock3", CRBlock(channel_size=5, original_channel_size=self.n_channels))
        ])
        self.feature_decoder_layers = nn.Sequential(feature_decoder)
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.feature_decoder_upsampling_layers(x)
        out = self.feature_decoder_layers(out)
        out = self.sigmoid(out)

        return out



if __name__ == "__main__":
    # random data
    x = np.random.random_sample((200, 12, 8, 8))
    x = torch.tensor(x).float()
    config = {
        "img_shape" : (2,32,32),
        "n_codeword_bits" :32,
        "embedding_dim" :8,
        "n_embeddings":16
    }

    # test encoder
    dec = Decoder(config=config)#, device=torch.device("cpu"))
    output = dec(x)
    print('Encoder out shape:', output.shape)
