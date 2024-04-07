# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.ReLU6)):
            pass
        else:
            m.initialize()


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation, bias=bias)


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=bias)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


# mobilenetv2.pth.tar
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()

        self.stride = stride

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# mobilenet_v2.pth.tar
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],          # 16, x / 2
            [6, 24, 2, 2],          # 24, x / 4
            [6, 32, 3, 2],          # 32, x / 8
            [6, 64, 4, 2],          # 64, x / 16
            [6, 96, 3, 1],          # 96, x / 16
            [6, 160, 3, 2],         # 160, x / 32
            #[6, 320, 1, 1],         # 320, x / 32
        ]

        # building first layer
        input_channel = int(32 * width_mult)
        self.features = [conv_bn(3, input_channel, 2)]

        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(InvertedResidual(input_channel, output_channel, s, t))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, t))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.initialize()

    def forward(self, x):
        layers = []
        for i, module in enumerate(self.features):
            x = module(x)
            if i in [3, 6, 10, 13, 16]:
                layers.append(x)
        return layers

    def initialize(self):
        self.load_state_dict(torch.load('../res/mobilenet_v2.pth.tar'), strict=False)


# Basic Fusion Module
class BFM(nn.Module):
    def __init__(self, channel):
        super(BFM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Feature Fusion Module
class FFM(nn.Module):
    def __init__(self, channel):
        super(FFM, self).__init__()
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_2):
        out = x_1 * x_2
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Cross Aggregation Module
class CAM(nn.Module):
    def __init__(self, channel):
        super(CAM, self).__init__()
        self.down = nn.Sequential(
            conv3x3(channel, channel, stride=2),
            nn.BatchNorm2d(channel)
        )
        self.conv_1 = conv3x3(channel, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)
        self.mul = FFM(channel)

    def forward(self, x_high, x_low):
        left_1 = x_low
        left_2 = F.relu(self.down(x_low), inplace=True)
        right_1 = F.interpolate(x_high, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        right_2 = x_high
        left = F.relu(self.bn_1(self.conv_1(left_1 * right_1)), inplace=True)
        right = F.relu(self.bn_2(self.conv_2(left_2 * right_2)), inplace=True)
        right = F.interpolate(right, size=x_low.size()[2:], mode='bilinear', align_corners=True)
        out = self.mul(left, right)
        return out

    def initialize(self):
        weight_init(self)


# Spatial Attention Module
class SAM(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(SAM, self).__init__()
        self.conv_atten = nn.Conv2d(2, 1, 5, padding=2, bias=False)
        self.conv = conv3x3(in_chan, out_chan)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        atten = torch.cat([avg_out, max_out], dim=1)
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten)
        out = F.relu(self.bn(self.conv(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Boundary Refinement Module
class BRM(nn.Module):
    def __init__(self, channel):
        super(BRM, self).__init__()
        self.conv_atten = conv1x1(channel*2, channel*2)
        self.conv_1 = conv3x3(channel*2, channel)
        self.bn_1 = nn.BatchNorm2d(channel)
        self.conv_2 = conv3x3(channel, channel)
        self.bn_2 = nn.BatchNorm2d(channel)

    def forward(self, x_1, x_edge):
        x = torch.cat([x_1, x_edge], dim=1)
        # x = x_1 + x_edge
        atten = F.avg_pool2d(x, x.size()[2:])
        atten = torch.sigmoid(self.conv_atten(atten))
        out = torch.mul(x, atten)
        out = F.relu(self.bn_1(self.conv_1(out) + x_1), inplace=True)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# Scale Adaptive Pooling Module
class SAP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SAP, self).__init__()
        self.conv_1 = conv3x3(in_channel, in_channel//4)
        self.bn_1 = nn.BatchNorm2d(in_channel//4)
        self.conv_2 = conv1x1(in_channel, out_channel)
        self.bn_2 = nn.BatchNorm2d(out_channel)

    def forward(self, x1, x2=None):
        if x2 != None:
            out = torch.cat([x1, x2], dim=1)
        else:
            out = x1
        out = F.relu(self.bn_1(self.conv_1(out)), inplace=True)
        out_1 = F.avg_pool2d(out, kernel_size=3, stride=1, padding=1)
        out_2 = F.avg_pool2d(out, kernel_size=5, stride=1, padding=2)
        out_3 = F.avg_pool2d(out, kernel_size=7, stride=1, padding=3)
        # out = out + out_1 + out_2 + out_3
        out = torch.cat([out, out_1, out_2, out_3], dim=1)
        out = F.relu(self.bn_2(self.conv_2(out)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


class CTDNet(nn.Module):
    def __init__(self, cfg):
        super(CTDNet, self).__init__()
        self.cfg = cfg
        self.bkbone = MobileNetV2(1.0)


        self.path1_1 = nn.Sequential(
            conv1x1(160, 64),
            nn.BatchNorm2d(64)
        )

        self.path1_2 = nn.Sequential(
            conv1x1(160, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.path1_3 = nn.Sequential(
            conv1x1(96+64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.path2 = SAM(32, 64)


        self.path3 = nn.Sequential(
            conv1x1(24, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.fuse1_1 = FFM(64)
        self.fuse1_2 = FFM(64)
        self.fuse12 = BFM(64)
        self.fuse3 = FFM(64)
        self.fuse23 = BRM(64)

        self.head_1 = conv3x3(64, 1, bias=True)
        self.head_2 = conv3x3(64, 1, bias=True)
        self.head_3 = conv3x3(64, 1, bias=True)
        self.head_4 = conv3x3(64, 1, bias=True)
        self.head_5 = conv3x3(64, 1, bias=True)
        self.head_edge = conv3x3(64, 1, bias=True)

        self.initialize()

    def forward(self, x, shape=None):
        shape = x.size()[2:] if shape is None else shape
        l3, l4, l5, l6, l7 = self.bkbone(x)

        path1_1 = F.avg_pool2d(l7, l7.size()[2:])
        path1_1 = self.path1_1(path1_1)
        path1_1 = F.interpolate(path1_1, size=l7.size()[2:], mode='bilinear', align_corners=True)   # 1/32
        path1_2 = self.path1_2(l7)                                                                  # 1/32
        path1_2 = self.fuse1_1(path1_1, path1_2)                                                    # 1/32
        path1_2 = F.interpolate(path1_2, size=l6.size()[2:], mode='bilinear', align_corners=True)   # 1/16

        path1_3 = self.path1_3(torch.cat([l6, l5], dim=1))                                          # 1/16
        path1 = self.fuse1_2(path1_2, path1_3)                                                      # 1/16
        path1 = F.interpolate(path1, size=l4.size()[2:], mode='bilinear', align_corners=True)

        path2 = self.path2(l4)                                                # 1/8
        path12 = self.fuse12(path1, path2)                                                          # 1/8
        path12 = F.interpolate(path12, size=l3.size()[2:], mode='bilinear', align_corners=True)     # 1/4

        path3_1 = self.path3(l3)                                              # 1/4
        path3_2 = F.interpolate(path1_2, size=l3.size()[2:], mode='bilinear', align_corners=True)   # 1/4
        path3 = self.fuse3(path3_1, path3_2)                                                        # 1/4

        path_out = self.fuse23(path12, path3)                                                       # 1/4

        logits_1 = F.interpolate(self.head_1(path_out), size=shape, mode='bilinear', align_corners=True)
        logits_edge = F.interpolate(self.head_edge(path3), size=shape, mode='bilinear', align_corners=True)

        if self.cfg.mode == 'train':
            logits_2 = F.interpolate(self.head_2(path12), size=shape, mode='bilinear', align_corners=True)
            logits_3 = F.interpolate(self.head_3(path1), size=shape, mode='bilinear', align_corners=True)
            logits_4 = F.interpolate(self.head_4(path1_2), size=shape, mode='bilinear', align_corners=True)
            logits_5 = F.interpolate(self.head_5(path1_1), size=shape, mode='bilinear', align_corners=True)
            return logits_1, logits_edge, logits_2, logits_3, logits_4, logits_5
        else:
            return logits_1, logits_edge

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)