import unittest
import torch
from model.centernet_head import CenternetHead


class MyTestCase(unittest.TestCase):
    def test_something(self):
        inputs = torch.rand(2, 64, 10, 10)
        layer = CenternetHead()
        out = layer(inputs)
        for k, v in out.items():
            print(f"{k}: {v.shape}")


if __name__ == '__main__':
    unittest.main()
