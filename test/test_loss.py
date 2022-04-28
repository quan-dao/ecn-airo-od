import unittest
import torch
from model.centernet_loss import focal_loss


class MyTestCase(unittest.TestCase):
    def test_focal_loss(self):
        sample_dict = torch.load('sample_target_dict.pth', map_location=torch.device('cpu'))
        target_dict = sample_dict['target_dict']
        pred_dict = {
            'cls': target_dict['target_cls'],
            'wh': target_dict['target_wh'],
            'reg': target_dict['target_offset']
        }
        loss = focal_loss(pred_dict['cls'], target_dict['target_cls'], True)
        print(f"loss: {loss}")


if __name__ == '__main__':
    unittest.main()
