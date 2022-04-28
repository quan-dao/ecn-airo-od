import unittest
import torch
from model.centernet_deconv import CenternetDeconv


class MyTestCase(unittest.TestCase):
    def test_forward(self):
        channels = [512, 256, 128, 64]
        neck = CenternetDeconv(channels)
        neck.cuda()
        print(neck)
        inputs = torch.rand(2, 512, 16, 16).float().cuda()
        output = neck(inputs)
        output_shape = (1, 2, 3, 4)  # TODO: compute output size
        self.assertTrue(output_shape == output.shape)


if __name__ == '__main__':
    unittest.main()
