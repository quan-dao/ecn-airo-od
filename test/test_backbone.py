import unittest
import torch
from model.backbone_resnet import ResnetBackbone


class MyTestCase(unittest.TestCase):
    def test_forward_pass(self):
        image = torch.rand(2, 3, 512, 512)
        backbone = ResnetBackbone()
        print(backbone)
        out = backbone(image)
        print(f"out: {out.shape}")
        out_shape = (1, 2, 3, 4)    # TODO: calculate size of out
        self.assertTrue(out_shape == out.shape)


if __name__ == '__main__':
    unittest.main()
