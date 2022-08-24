import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from .mobilenet_v2 import mobilenet_v2


class MobileNetV2(nn.Module):
    """MobileNet Body"""

    def __init__(self, pretrained=False):
        super(MobileNetV2, self).__init__()
        self.model = mobilenet_v2(pretrained=pretrained)

    def forward(self, x):
        out3 = self.model.features[:7](x)
        out4 = self.model.features[7:14](out3)
        out5 = self.model.features[14:18](out4)
        return out3, out4, out5


def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    """Normal Conv"""
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size,
         stride=stride, padding=pad, groups=groups, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
        # ("relu", nn.ReLU(inplace=True)),
    ]))


def conv_dw(filter_in, filter_out, stride=1):
    """DW Conv"""
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride,
                  1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),

        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )


class SpatialPyramidPooling(nn.Module):
    """SPP"""

    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList(
            [nn.MaxPool2d(pool_size, 1, pool_size//2) for pool_size in pool_sizes])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


class Upsample(nn.Module):
    """上采样"""

    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


def make_three_conv(filters_list, in_filters):
    """三组卷积"""
    ret = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),
        conv_dw(filters_list[0], filters_list[1]),
        conv2d(filters_list[1], filters_list[0], 1),
    )
    return ret


def make_five_conv(filters_list, in_filters):
    """五组卷积"""
    ret = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1),  # 维度变换
        conv_dw(filters_list[0], filters_list[1]),  # 扩展
        conv2d(filters_list[1], filters_list[0], 1),  # 提取特征
        conv_dw(filters_list[0], filters_list[1]),  # 还原
        conv2d(filters_list[1], filters_list[0], 1),  # 输出变换
    )
    return ret


def yolo_head(filters_list, in_filters):
    """Yolo Head输出"""
    ret = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return ret


class YoloBody(nn.Module):
    """Yolo Body"""

    def __init__(self, anchors_mask, num_classes, backbone="mobilenetv2", pretrained=False):
        super(YoloBody, self).__init__()
        # 生成MobileNet主干
        if backbone == "mobilenetv2":
            # x2: 52,52,32
            # x1: 26,26,92
            # x0: 13,13,320
            self.backbone = MobileNetV2(pretrained=pretrained)
        else:  # error
            raise ValueError(
                'Unsupported backbone - `{}`, Use mobilenetv2 only.'.format(backbone))

        self.conv1 = make_three_conv([512, 1024], 320)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = conv2d(96, 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        # self.conv_for_P3 = conv2d(in_filters[0], 128, 1)
        self.conv_for_P3 = conv2d(32, 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        self.down_sample1 = conv_dw(128, 256, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)

        self.down_sample2 = conv_dw(256, 512, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)

        # 3*(5+num_classes)=3*(5+20)=3*(4+1+20)=75
        self.yolo_head1 = yolo_head(
            [1024, len(anchors_mask[2]) * (5 + num_classes)], 512)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head2 = yolo_head(
            [512, len(anchors_mask[1]) * (5 + num_classes)], 256)

        # 3*(5+num_classes) = 3*(5+20) = 3*(4+1+20)=75
        self.yolo_head3 = yolo_head(
            [256, len(anchors_mask[0]) * (5 + num_classes)], 128)

    def forward(self, x):
        """backbone"""
        x2, x1, x0 = self.backbone(x)  # MobileNet Outputs
        """上采样金字塔开始"""
        # 13,13,1024
        # 13,13,512
        # 13,13,1024
        # 13,13,512
        # 13,13,2048
        P5 = self.conv1(x0)  # 获取x0后进行五组卷积

        P5 = self.SPP(P5)  # SPP

        # 13,13,2048
        # 13,13,512
        # 13,13,1024
        # 13,13,512
        P5 = self.conv2(P5)  # 三组卷积

        # 13,13,512
        # 13,13,256
        # 26,26,256
        P5_upsample = self.upsample1(P5) # 上采样

        # 26,26,512
        # 26,26,256
        P4 = self.conv_for_P4(x1) # 获取x1
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P4, P5_upsample], axis=1) # 合并x1和x2
        # 26,26,512
        # 26,26,256
        # 26,26,512
        # 26,26,256
        # 26,26,512
        # 26,26,256
        P4 = self.make_five_conv1(P4) # 获取x1并与x0特征融合后(得到x1')进行五组卷积

        # 26,26,256
        # 26,26,128
        # 52,52,128
        P4_upsample = self.upsample2(P4) # 上采样

        # 52,52,256
        # 52,52,128
        P3 = self.conv_for_P3(x2) # 获取x2
        # 52,52,128 + 52,52,128 -> 52,52,256
        P3 = torch.cat([P3, P4_upsample], axis=1) # 合并x2和x1'
        """上采样金字塔结束"""
        """下采样金字塔开始"""
        # 52,52,256
        # 52,52,128
        # 52,52,256
        # 52,52,128
        # 52,52,256
        # 52,52,128
        P3 = self.make_five_conv2(P3) # 获取x2并与x1'特征融合后(得到x2')进行五组卷积

        # 52,52,128
        # 26,26,256
        P3_downsample = self.down_sample1(P3) # 整体下采样
        # 26,26,256 + 26,26,256 -> 26,26,512
        P4 = torch.cat([P3_downsample, P4], axis=1) # 合并x2'的下采样图和x1'

        # 26,26,512
        # 26,26,256
        # 26,26,512
        # 26,26,256
        # 26,26,512
        # 26,26,256
        P4 = self.make_five_conv3(P4) # 进行五组卷积

        # 26,26,256
        # 13,13,512
        P4_downsample = self.down_sample2(P4) # 整体下采样
        # 13,13,512 + 13,13,512 -> 13,13,1024
        P5 = torch.cat([P4_downsample, P5], axis=1) # 合并x1'的下采样图和x0'
        # 13,13,1024
        # 13,13,512
        # 13,13,1024
        # 13,13,512
        # 13,13,1024
        # 13,13,512
        P5 = self.make_five_conv4(P5) # 进行五组卷积
        """下采样金字塔结束"""
        """输出yolo head"""
        # 第三个特征层
        out2 = self.yolo_head3(P3) # y3=(batch_size,75,52,52)
        # 第二个特征层
        out1 = self.yolo_head2(P4) # y2=(batch_size,75,26,26)
        # 第一个特征层
        out0 = self.yolo_head1(P5) # y1=(batch_size,75,13,13)

        return out0, out1, out2
