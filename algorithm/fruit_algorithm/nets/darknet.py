import torch.nn as nn
import math
from collections import OrderedDict


class ResUnit(nn.Module):
    def __init__(self, inplanes, planes):
        super(ResUnit, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(
            planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        # DBL_1
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        # DBL_2
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        # ADD
        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # DBL
        # 416,416,3 -> 416,416,32
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)
        # res1
        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0])
        # res2
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])
        # res8
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])
        # res8
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])
        # res4
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024] # out_filters : [64, 128, 256, 512, 1024]

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):
        # Generate Resnet Blocks
        layers = []
        # DownSample
        layers.append(("dbl_conv", nn.Conv2d(
            self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("dbl_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("dbl_relu", nn.LeakyReLU(0.1)))
        # ResUnit
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(
                i), ResUnit(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # DBL
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # res1
        x = self.layer2(x)  # res2
        out3 = self.layer3(x)  # res8
        out4 = self.layer4(out3)  # res8
        out5 = self.layer5(out4)  # res4

        return out3, out4, out5

def generate_darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model