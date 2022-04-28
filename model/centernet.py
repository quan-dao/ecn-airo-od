import math
import numpy as np
import torch
import torch.nn as nn
from model.backbone_resnet import ResnetBackbone
from model.centernet_deconv import CenternetDeconv
from model.centernet_head import CenternetHead


class CenterNet(nn.Module):
    def __init__(self, use_cuda=True):
        super().__init__()
        self.device = torch.device('cuda') if use_cuda else torch.device('cpu')
        self.down_scale = 4
        self.backbone = ResnetBackbone()
        self.upsample = CenternetDeconv([512, 256, 128, 64])  # TODO
        self.head = CenternetHead()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        pixel_mean = torch.tensor(self.mean).float().to(self.device).view(1, 3, 1, 1)
        pixel_std = torch.tensor(self.std).float().to(self.device).view(1, 3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def forward(self, inputs):
        pass

    @torch.no_grad()
    def inference(self, images: torch.Tensor):
        """
        image: (N, C, H, W)
        """
        n, c, h, w = images.shape
        new_h, new_w = (h | 31) + 1, (w | 31) + 1
        center_wh = np.array([w // 2, h // 2], dtype=np.float32)
        size_wh = np.array([new_w, new_h], dtype=np.float32)
        down_scale = self.down_scale
        img_info = dict(center=center_wh, size=size_wh,
                        height=new_h // down_scale,
                        width=new_w // down_scale)

        pad_value = [-x / y for x, y in zip(self.mean, self.std)]
        aligned_img = torch.Tensor(pad_value).reshape((1, -1, 1, 1)).expand(n, c, new_h, new_w)
        aligned_img = aligned_img.to(self.device)

        pad_w, pad_h = math.ceil((new_w - w) / 2), math.ceil((new_h - h) / 2)
        aligned_img[..., pad_h:h + pad_h, pad_w:w + pad_w] = images

        features = self.backbone(aligned_img)
        up_fmap = self.upsample(features)
        pred_dict = self.head(up_fmap)
        return pred_dict

    def preprocess_image(self, batched_inputs: torch.Tensor):
        """
        Normalize, pad and batch the input images.
        Args:
            batched_inputs: (N, 3, H, W)
        """
        images = self.normalizer(batched_inputs / 255.0)
        return images
