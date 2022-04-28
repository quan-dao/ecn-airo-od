import unittest
from dataset.coco_dataset import COCODataset, create_coco_loader
import matplotlib.pyplot as plt


class MyTestcase(unittest.TestCase):
    def test_getitem(self):
        coco_dataset = COCODataset()
        data_dict = coco_dataset[15]
        coco_dataset.draw_annotation(data_dict['image'], data_dict['gt_boxes'])
        fig, axe = plt.subplots()
        axe.imshow(data_dict['image'][..., ::-1])
        plt.show()

    def test_collate(self):
        coco_dataset = COCODataset()
        batch_size = 2
        coco_loader = create_coco_loader(coco_dataset, batch_size)
        coco_loader_iter = iter(coco_loader)
        batch_dict = next(coco_loader_iter)
        for i in range(batch_size):
            coco_dataset.draw_annotation(batch_dict['images'][i], batch_dict['gt_boxes'][i])

        fig, axe = plt.subplots(1, batch_size)
        for i in range(batch_size):
            axe[i].imshow(batch_dict['images'][i, :, :, ::-1])
        plt.show()


