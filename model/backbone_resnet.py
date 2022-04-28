"""
src: https://github.com/FateScript/CenterNet-better
"""
import torch.nn as nn
import torchvision.models.resnet as resnet


class ResnetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet.resnet18(pretrained=False)
        self.stage0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
