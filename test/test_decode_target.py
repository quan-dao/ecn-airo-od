import unittest
import torch
import cv2
import matplotlib.pyplot as plt
from dataset.coco_dataset import COCODataset
from model.target_decoder import decode_centernet_target


class MyTestCase(unittest.TestCase):
    def test_decode_ground_truth(self):
        coco_dataset = COCODataset()
        sample_dict = torch.load('sample_target_dict.pth', map_location=torch.device('cpu'))
        target_dict = sample_dict['target_dict']
        batch_dict = sample_dict['batch_dict']
        # clone target_dict to make pred_dict
        pred_dict = {
            'cls': target_dict['target_cls'],
            'wh': target_dict['target_wh'],
            'reg': target_dict['target_offset']
        }
        batch_size = pred_dict['cls'].shape[0]
        decoded_boxes = decode_centernet_target(pred_dict, output_stride=1, cls_is_activated=True, max_num_obj=100)
        print(f"decoded_boxes: {decoded_boxes.shape}")
        mask_above_thresh = decoded_boxes[..., -1] > 0.7  # (B, n_boxes)
        decoded_boxes = [decoded_boxes[i, mask_above_thresh[i]].numpy() for i in range(batch_size)]

        # vis
        images = batch_dict['images']
        fig, axe = plt.subplots(1, batch_size)
        for batch_idx in range(batch_size):
            img = images[batch_idx]
            cur_boxes = decoded_boxes[batch_idx]
            coco_dataset.draw_annotation(img, cur_boxes)
            axe[batch_idx].imshow(img[..., ::-1])
        plt.show()

    def test_decode_prediction(self):
        coco_dataset = COCODataset()
        img_name = '000000000885'
        img = cv2.imread(f'../data/val2017/{img_name}.jpg')  # (H, W, 3)
        img = cv2.resize(img, coco_dataset.input_size, interpolation=cv2.INTER_CUBIC)
        pred_dict = torch.load(f'pred_for_{img_name}.pth', map_location=torch.device('cpu'))
        for k, v in pred_dict.items():
            print(f"{k}: {v.shape}")
        decoded_boxes = decode_centernet_target(pred_dict, output_stride=4, cls_is_activated=True, max_num_obj=100)
        print(f"decoded_boxes: {decoded_boxes.shape}")
        mask_above_thresh = decoded_boxes[..., -1] > 0.3  # (B, n_boxes)
        decoded_boxes = decoded_boxes[0, mask_above_thresh[0]].numpy()
        print(f"decoded_boxes ({decoded_boxes.shape}):\n{decoded_boxes}")
        fig, axe = plt.subplots()
        coco_dataset.draw_annotation(img, decoded_boxes)
        axe.imshow(img[..., ::-1])
        plt.show()


if __name__ == '__main__':
    unittest.main()
