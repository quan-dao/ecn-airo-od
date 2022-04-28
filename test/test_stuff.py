import unittest
from dataset.coco_dataset import COCODataset


class MyTestCase(unittest.TestCase):
    def test_something(self):
        coco_dataset = COCODataset()
        print(coco_dataset.cls_names)
        print(coco_dataset.image_ids[:10])
        data = coco_dataset[10]
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
