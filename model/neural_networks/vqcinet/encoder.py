import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
References:

Lu, Zhilin, Jintao Wang, and Jian Song. "Multi-resolution CSI feedback with deep learning in massive MIMO system." ICC 2020-2020 IEEE International Conference on Communications (ICC). IEEE, 2020.

https://github.com/Kylin9511/CRNet/blob/master/main.py
"""

import torch
import torch.nn as nn
from collections import OrderedDict


class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, groups=1, NOPADDING=False):
        if not isinstance(kernel_size, int):
            padding = [(i - 1) // 2 for i in kernel_size]
        else:
            padding = (kernel_size - 1) // 2
        if NOPADDING:
            super(ConvBN, self).__init__(OrderedDict([
                ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                   groups=groups, bias=False)),
                ('bn', nn.BatchNorm2d(out_planes))
            ]))
        else:
            super(ConvBN, self).__init__(OrderedDict([
                ('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                                   padding=padding, groups=groups, bias=False)),
                ('bn', nn.BatchNorm2d(out_planes))
            ]))
class Encoder(nn.Module):

    def __init__(self,  img_shape):
        super(Encoder, self).__init__()
        self.img_shape = img_shape
        self.n_channels = self.img_shape[0]
        self.img_size = self.img_shape[0] * self.img_shape[1] * self.img_shape[2]
        #self.codeword_length = config["n_codeword_floats"]
        #reduction = self.img_size // self.codeword_length
        self.feature_encoder1 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(self.n_channels, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 9])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [9, 1])),
        ]))
        self.feature_encoder3 = nn.Sequential(OrderedDict([
            ("conv3x3_bn", ConvBN(self.n_channels, 2, 3)),
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv1x9_bn", ConvBN(2, 2, [1, 3])),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ("conv9x1_bn", ConvBN(2, 2, [3, 1])),
        ]))
        self.feature_encoder2 = ConvBN(self.n_channels, 2, 3)
        # self.feature_encoder_conv = nn.Sequential(OrderedDict([
        #     ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        #     ("conv1x1_bn", ConvBN(6, self.n_channels, 1)),
        #     ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        # ]))
        self.feature_encoder_conv = nn.Sequential(OrderedDict([
            ("relu1", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv1', nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(2,2), stride=(2,2), bias=False)),
            ('bn1', nn.BatchNorm2d(12)),
            ("relu2", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
            ('conv2', nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(2,2), stride=(2,2), bias=False)),
            ('bn2', nn.BatchNorm2d(12)),
            ("relu3", nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ]))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        #n, c, h, w = x.detach().size()
        encode1 = self.feature_encoder1(x)
        encode2 = self.feature_encoder2(x)
        encode3 = self.feature_encoder3(x)
        out = torch.cat((encode1, encode2, encode3), dim=1)
        out = self.feature_encoder_conv(out)

        return out
        # out = self.encoder_fc(out.view(n, -1))


if __name__ == "__main__":
    # random data
    x = np.random.random_sample((1, 2, 32, 32))
    x = torch.tensor(x).float()
    config = {
        "img_shape" : (2,32,32),
        "n_codeword_floats" : 32
    }

    # test encoder
    encoder = Encoder(config=config)#, device=torch.device("cuda:"+str(0) if torch.cuda.is_available() and 0>-1 else "cpu"))
    start_time = time.time()
    encoder_out = encoder(x)
    end_time = time.time()
    print('Encoder out shape:', encoder_out.shape, "takes {} sec.".format(end_time-start_time))

