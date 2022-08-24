from turtle import back
import torch
import torch.nn as nn

from collections import OrderedDict

from nets.darknet import generate_darknet53

# def conv2d(filter_in, filter_out, kernel_size):
#     # NormalConv
#     pad = (kernel_size - 1) // 2 if kernel_size else 0
#     return nn.Sequential(OrderedDict([
#         ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
#         ("bn", nn.BatchNorm2d(filter_out)),
#         ("relu", nn.LeakyReLU(0.1)),
#     ]))

def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.ReLU6(inplace=True)),
    ]))

def make_last_layers(filters_list, in_filters, out_filter):
    # Generate Last Layer
    m = nn.Sequential(
        conv2d(in_filters, filters_list[0], 1), # get feature 1
        conv2d(filters_list[0], filters_list[1], 3), # get feature 2
        conv2d(filters_list[1], filters_list[0], 1), # get feature 3
        conv2d(filters_list[0], filters_list[1], 3), # get feature 4
        conv2d(filters_list[1], filters_list[0], 1), # get feature 5
        conv2d(filters_list[0], filters_list[1], 3), # get result 1
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True) # get result 2
    )
    return m

class Yolo(nn.Module):
    def __init__(self, anchors_mask, num_classes, backbone="darknet53", pretrained = False, pretained_path = None):
        super(Yolo, self).__init__()
        if backbone == "darknet53":
            self.backbone = generate_darknet53()
        elif backbone == "mobilenetv2":
            # 52,52,32;26,26,92;13,13,320
            self.backbone = generate_mobilenetv2(pretrained=pretrained)
            in_filters = [32, 96, 320]
        elif backbone == "mobilenev3":
            # 52,52,40;26,26,112;13,13,160
            self.backbone = generate_mobilenetv3(pretrained=pretrained)
            in_filters = [40, 112, 160]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use darknet53, mobilenetv2, mobilenetv3.'.format(backbone))

        """
        Get Features in Shape of:
        52,52,256 max
        26,26,512 mid
        13,13,1024 min
        """
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretained_path))

        out_filters = self.backbone.layers_out_filters # out_filters : [64, 128, 256, 512, 1024]

        """
        FOR VOC DATASET!!!
        Calculate the output channels of yolo_head
        final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        """
        self.last_layer0            = make_last_layers([512, 1024], out_filters[-1], len(anchors_mask[0]) * (num_classes + 5))

        self.last_layer1_conv       = conv2d(512, 256, 1)
        self.last_layer1_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1            = make_last_layers([256, 512], out_filters[-2] + 256, len(anchors_mask[1]) * (num_classes + 5))

        self.last_layer2_conv       = conv2d(256, 128, 1)
        self.last_layer2_upsample   = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2            = make_last_layers([128, 256], out_filters[-3] + 128, len(anchors_mask[2]) * (num_classes + 5))

    def forward(self, x):
        # Get Features in Shape of:
        # 52,52,256 max
        # 26,26,512 mid
        # 13,13,1024 min
        x2, x1, x0 = self.backbone(x)

        """
        Feature3 from out5
        13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        """
        # Get Branch
        out0_branch = self.last_layer0[:5](x0)
        out0        = self.last_layer0[5:](out0_branch) # out0 = (batch_size,255,13,13)
        # UpSample
        # 13,13,512 -> 13,13,256 -> 26,26,256
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        # Concat
        # 26,26,256 + 26,26,512 -> 26,26,768
        x1_in = torch.cat([x1_in, x1], 1)

        """
        Feature2 from out4
        26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        """
        # Get Branch
        out1_branch = self.last_layer1[:5](x1_in)
        out1        = self.last_layer1[5:](out1_branch) # out1 = (batch_size,255,26,26)
        # UpSample
        # 26,26,256 -> 26,26,128 -> 52,52,128
        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        # Concat
        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)

        """
        Feature1 from out3
        52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        """
        out2 = self.last_layer2(x2_in)

        # Output
        return out0, out1, out2