"""
src: https://github.com/FateScript/CenterNet-better
"""
import torch
import torch.nn as nn


class SingleHead(nn.Module):

    def __init__(self, in_channel, out_channel, bias_fill=False, bias_value=0.):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class CenternetHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, input_channels=64, num_classes=80):
        super(CenternetHead, self).__init__()
        self.cls_head = SingleHead(
            input_channels,
            num_classes,
            bias_fill=True,
            bias_value=-2.19,
        )
        self.wh_head = SingleHead(input_channels, 2)
        self.reg_head = SingleHead(input_channels, 2)

    def forward(self, x):
        cls = self.cls_head(x)
        cls = torch.sigmoid(cls)
        wh = self.wh_head(x)
        reg = self.reg_head(x)
        pred = {
            'cls': cls,
            'wh': wh,
            'reg': reg
        }
        return pred
