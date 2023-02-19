#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Created on 2023/02/18

@author: drumichiro
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardSwishBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, groups=1):
        super(HardSwishBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              groups=groups, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        h = F.relu6(x + 3.0) / 6.0
        return x * h


class AvePoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ave_pool_size):
        super(AvePoolBlock, self).__init__()
        self.average_pool = nn.AvgPool2d(kernel_size=ave_pool_size,
                                         stride=ave_pool_size, padding=0)
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=in_channels,
                               kernel_size=1, stride=1)

    def forward(self, x, h):
        h = self.average_pool(h)
        h = F.relu(self.conv1(h))
        h = torch.sigmoid(self.conv2(h))
        return x * h


class FinalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, interp_size, avepool_block):
        super(FinalBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               groups=out_channels, padding=1)
        self.interp = nn.Upsample(
            size=interp_size,
            mode="bilinear",
            align_corners=False)
        self.avepool_block = avepool_block

    def forward(self, x, h):
        x = self.interp(x)
        h1 = self.conv1(x)
        h2 = self.avepool_block(h, h + h1)
        x = h2 + h1
        h1 = F.relu(self.conv2(x))
        h2 = F.relu(self.conv3(h1))
        return h1 + h2


class SelfieSegmentation(nn.Module):
    def __init__(self):
        super(SelfieSegmentation, self).__init__()
        self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)
        self.conv2_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, groups=16)
        self.conv2_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1)
        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=72, kernel_size=1, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=72, out_channels=72, kernel_size=3, stride=2, groups=72)
        self.conv3_3 = nn.Conv2d(in_channels=72, out_channels=24, kernel_size=1, stride=1)
        self.conv4_1 = nn.Conv2d(in_channels=24, out_channels=88, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=88, out_channels=88, kernel_size=3, stride=1, padding=1, groups=88)
        self.conv4_3 = nn.Conv2d(in_channels=88, out_channels=24, kernel_size=1, stride=1)

        self.conv5 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=1, stride=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1)
        self.conv7_1 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1)
        self.conv8 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=1, stride=1)
        self.conv9 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=1, stride=1)
        self.conv10_1 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1)
        self.conv10_2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=1, stride=1)

        self.hswish_block1 = HardSwishBlock(in_channels=3, out_channels=16, kernel_size=3, stride=2)
        self.hswish_block5_1 = HardSwishBlock(in_channels=24, out_channels=96, kernel_size=1, stride=1)
        self.hswish_block5_2 = HardSwishBlock(in_channels=96, out_channels=96, kernel_size=5, stride=2, groups=96)
        self.hswish_block6_1 = HardSwishBlock(in_channels=32, out_channels=128, kernel_size=1, stride=1)
        self.hswish_block6_2 = HardSwishBlock(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, groups=128)
        self.hswish_block7_1 = HardSwishBlock(in_channels=32, out_channels=128, kernel_size=1, stride=1)
        self.hswish_block7_2 = HardSwishBlock(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2, groups=128)
        self.hswish_block8_1 = HardSwishBlock(in_channels=32, out_channels=96, kernel_size=1, stride=1)
        self.hswish_block8_2 = HardSwishBlock(in_channels=96, out_channels=96, kernel_size=5, stride=1, padding=2, groups=96)
        self.hswish_block9_1 = HardSwishBlock(in_channels=32, out_channels=96, kernel_size=1, stride=1)
        self.hswish_block9_2 = HardSwishBlock(in_channels=96, out_channels=96, kernel_size=5, stride=1, padding=2, groups=96)

        self.avepool_block1 = AvePoolBlock(in_channels=16, out_channels=8, ave_pool_size=64)
        self.avepool_block5 = AvePoolBlock(in_channels=96, out_channels=24, ave_pool_size=16)
        self.avepool_block6 = AvePoolBlock(in_channels=128, out_channels=32, ave_pool_size=16)
        self.avepool_block7 = AvePoolBlock(in_channels=128, out_channels=32, ave_pool_size=16)
        self.avepool_block8 = AvePoolBlock(in_channels=96, out_channels=24, ave_pool_size=16)
        self.avepool_block9 = AvePoolBlock(in_channels=96, out_channels=24, ave_pool_size=16)

        self.final_block1 = FinalBlock(in_channels=128, out_channels=24, interp_size=[32, 32],
                                       avepool_block=AvePoolBlock(in_channels=24, out_channels=24, ave_pool_size=32))
        self.final_block2 = FinalBlock(in_channels=24, out_channels=16, interp_size=[64, 64],
                                       avepool_block=AvePoolBlock(in_channels=16, out_channels=16, ave_pool_size=64))
        self.final_block3 = FinalBlock(in_channels=16, out_channels=16, interp_size=[128, 128],
                                       avepool_block=AvePoolBlock(in_channels=16, out_channels=16, ave_pool_size=128))

        self.avepool10 = nn.AvgPool2d(kernel_size=16, stride=16, padding=0)
        self.conv_transpose = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        mul1 = self.hswish_block1(F.pad(x, (0, 1, 0, 1)))
        ######################
        h2 = F.relu(self.conv2_1(mul1))
        h2 = F.relu(self.conv2_2(F.pad(h2, (0, 1, 0, 1))))
        h2 = self.avepool_block1(h2, h2)
        cnv2 = self.conv2_3(h2)
        ######################
        h3 = F.relu(self.conv3_1(cnv2))
        h3 = F.relu(self.conv3_2(F.pad(h3, (0, 1, 0, 1))))
        cnv3 = self.conv3_3(h3)
        ######################
        h4 = F.relu(self.conv4_1(cnv3))
        h4 = F.relu(self.conv4_2(h4))
        h4 = self.conv4_3(h4)
        add4 = h4 + cnv3
        ######################
        h5 = self.hswish_block5_1(add4)
        h5 = self.hswish_block5_2(F.pad(h5, (1, 2, 1, 2)))
        h5 = self.avepool_block5(h5, h5)
        cnv5 = self.conv5(h5)
        ######################
        h6 = self.hswish_block6_1(cnv5)
        h6 = self.hswish_block6_2(h6)
        h6 = self.avepool_block6(h6, h6)
        h6 = self.conv6(h6)
        add6 = h6 + cnv5
        ######################
        h7 = self.hswish_block7_1(add6)
        h7 = self.hswish_block7_2(h7)
        h7 = self.avepool_block7(h7, h7)
        h7 = self.conv7_1(h7)
        add7 = h7 + add6
        ######################
        h8 = self.hswish_block8_1(add7)
        h8 = self.hswish_block8_2(h8)
        h8 = self.avepool_block8(h8, h8)
        h8 = self.conv8(h8)
        add8 = h8 + add7
        ######################
        h9 = self.hswish_block9_1(add8)
        h9 = self.hswish_block9_2(h9)
        h9 = self.avepool_block9(h9, h9)
        h9 = self.conv9(h9)
        add9 = h9 + add8
        ######################
        h10_l = self.avepool10(add9)
        h10_l = torch.sigmoid(self.conv10_1(h10_l))
        h10_r = F.relu(self.conv10_2(add9))
        mul10 = h10_r * h10_l
        ######################
        fin1 = self.final_block1(mul10, add4)
        fin2 = self.final_block2(fin1, cnv2)
        fin3 = self.final_block3(fin2, mul1)
        ######################
        return torch.sigmoid(self.conv_transpose(fin3) + 0.5327)


def main():
    import demo_static_image
    demo_static_image.main()


if __name__ == "__main__":
    main()
    print("Done.")
