#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
Created on 2023/05/06

@author: drumichiro
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, in_kernel_size=1, in_stride=1, in_padding=0, mid_stride=1, dilation=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=in_kernel_size, stride=in_stride, padding=in_padding)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, stride=mid_stride, padding=dilation, dilation=dilation, groups=mid_channels)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.add_if = torch.add if in_channels == out_channels else self.pass_through

    def pass_through(self, x, h):
        return h

    def forward(self, x):
        h = F.relu6(self.conv1(x))
        h = F.relu6(self.conv2(h))
        h = self.conv3(h)
        return self.add_if(x, h)


class DeepLabV3(nn.Module):
    def __init__(self):
        super(DeepLabV3, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=33, stride=33, padding=0)
        self.interp0 = nn.Upsample(size=[33, 33], mode="bilinear", align_corners=True)
        self.interp1 = nn.Upsample(size=[257, 257], mode="bilinear", align_corners=True)

        self.conv_block1 = ConvBlock(in_channels=3, out_channels=8, mid_channels=16, in_kernel_size=3, in_stride=2, in_padding=1)
        self.conv_block2 = ConvBlock(in_channels=8, out_channels=12, mid_channels=48, mid_stride=2)
        self.conv_block3 = ConvBlock(in_channels=12, out_channels=12, mid_channels=72)
        self.conv_block4 = ConvBlock(in_channels=12, out_channels=16, mid_channels=72, mid_stride=2)
        self.conv_block5 = ConvBlock(in_channels=16, out_channels=16, mid_channels=96)
        self.conv_block6 = ConvBlock(in_channels=16, out_channels=16, mid_channels=96)
        self.conv_block7 = ConvBlock(in_channels=16, out_channels=32, mid_channels=96)
        self.conv_block8 = ConvBlock(in_channels=32, out_channels=32, mid_channels=192, dilation=2)
        self.conv_block9 = ConvBlock(in_channels=32, out_channels=32, mid_channels=192, dilation=2)
        self.conv_block10 = ConvBlock(in_channels=32, out_channels=32, mid_channels=192, dilation=2)
        self.conv_block11 = ConvBlock(in_channels=32, out_channels=48, mid_channels=192, dilation=2)
        self.conv_block12 = ConvBlock(in_channels=48, out_channels=48, mid_channels=288, dilation=2)
        self.conv_block13 = ConvBlock(in_channels=48, out_channels=48, mid_channels=288, dilation=2)
        self.conv_block14 = ConvBlock(in_channels=48, out_channels=80, mid_channels=288, dilation=2)
        self.conv_block15 = ConvBlock(in_channels=80, out_channels=80, mid_channels=480, dilation=4)
        self.conv_block16 = ConvBlock(in_channels=80, out_channels=80, mid_channels=480, dilation=4)
        self.conv_block17 = ConvBlock(in_channels=80, out_channels=160, mid_channels=480, dilation=4)

        self.conv1 = nn.Conv2d(in_channels=160, out_channels=256, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=160, out_channels=256, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=21, kernel_size=1, stride=1)

    def forward(self, x):
        h1 = self.conv_block1(x)
        h1 = self.conv_block2(h1)
        h2 = self.conv_block3(h1)
        ######################
        h2 = self.conv_block4(h2)
        h3 = self.conv_block5(h2)
        ######################
        h4 = self.conv_block6(h3)
        ######################
        h4 = self.conv_block7(h4)
        h5 = self.conv_block8(h4)
        ######################
        h6 = self.conv_block9(h5)
        ######################
        h7 = self.conv_block10(h6)
        ######################
        h7 = self.conv_block11(h7)
        h8 = self.conv_block12(h7)
        ######################
        h9 = self.conv_block13(h8)
        ######################
        h9 = self.conv_block14(h9)
        h10 = self.conv_block15(h9)
        ######################
        h11 = self.conv_block16(h10)
        ######################
        h12 = self.conv_block17(h11)
        ######################
        h13 = F.relu(self.conv1(h12))
        h14 = self.interp0(F.relu(self.conv2(self.avg_pool(h12))))
        h14 = torch.cat([h14, h13], dim=1)
        h14 = F.relu(self.conv3(h14))
        h14 = self.interp0(self.conv4(h14))
        return self.interp1(h14)


def main():
    from demo_image_segmentation import illustrate
    illustrate("samples/family_usj_snw.jpg")


if __name__ == "__main__":
    main()
    print("Done.")
