import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os.path as osp
from pycocotools.coco import COCO
import cv2
from typing import Dict, List


class COCODataset(Dataset):
    def __init__(self, dataset_dir='../data', dataset_type='val2017', input_size=(512, 512)):
        ann_file = osp.join(dataset_dir, 'annotations', f"instances_{dataset_type}.json")
        assert osp.exists(ann_file), f"{ann_file} does not exist"
        self.images_dir = osp.join(dataset_dir, dataset_type)
        self.coco = COCO(ann_file)
        self.image_ids = self.coco.getImgIds()
        self.image_ids.sort()
        categories = self.coco.loadCats(self.coco.getCatIds())
        self.cls_names = [cat['name'] for cat in categories]
        category_ids = self.coco.getCatIds(catNms=self.cls_names)
        self.cat_id_to_name = dict(zip(category_ids, self.cls_names))
        self.cat_id_to_cls_idx = dict(zip(category_ids, range(len(self.cls_names))))
        self.name_to_cls_idx = dict(zip(self.cls_names, range(len(self.cls_names))))
        self.input_size = input_size  # (height, width)

    def __len__(self):
        return -1  # TODO

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        :param idx: index pointing to an image whose ID stored in self.image_ids
        :return: {
            'image': (H, W, 3),
            'gt_boxes': (N_gt, 5) - top_x, top_y, w, h, cls_idx
        }
        """
        info = self.coco.loadImgs(self.image_ids[idx])[0]
        out = dict()
        # get image
        img = cv2.imread(osp.join(self.images_dir, 'provide_file_name'))  # TODO:
        out['image'] = None  # TODO: use cv2.resize to resize img to the size of self.input_size

        # get annotations
        anno_ids = self.coco.getAnnIds(imgIds=info['id'], iscrowd=None)
        annos = self.coco.loadAnns(anno_ids)
        gt_boxes = []
        for ann in annos:
            bbox = ann['bbox']  # [upper_x, upper_y, w, h]
            resized_upper_x, resized_upper_y, resized_w, resized_h = [0] * 4  # TODO: resize bbox
            # TODO: (cont) reflect the fact that img is resized from its original size to self.input_size
            resized_bbox = [
                resized_upper_x, resized_upper_y,
                resized_w, resized_h,
                float(self.cat_id_to_cls_idx[ann['category_id']])
            ]
            gt_boxes.append(resized_bbox)
        out['gt_boxes'] = np.array(gt_boxes, dtype=float)  # (N_gt, 5)
        return out

    @staticmethod
    def collate(batch_data: List[Dict]) -> Dict:
        """
        :param batch_data: list of output of __getitem__
        :return: {
            'images': (B, H, W, 3),
            'gt_boxes': [(N0, 5), (N1, 5), ...]
        }
        """
        images, gt_boxes = [], []
        for data in batch_data:
            img = None  # TODO: access the image stored in data and prepend it with 1 dimension
            # TODO: (cont) (H, W, 3) -> (1, H, W, 3)
            images.append(img)
            boxes = None  # TODO: access gt_boxes stored in data & assign it to boxes
            gt_boxes.append(boxes)
        out = {
            'images': None,  # TODO: concatenate images along the 1st dimension using np.concatenate
            'gt_boxes': gt_boxes
        }
        return out

    def draw_annotation(self, img_: np.ndarray, gt_boxes: np.ndarray) -> None:
        """
        Draw gt_boxes over img_
        :param img_: (H, W, 3) - to be mutated
        :param gt_boxes: (N, 5) - top_x, top_y, w, h, cls_idx
        """
        assert len(img_.shape) == 3 and img_.shape[2] == 3, f"{img_.shape} != (H, W, 3)"
        for i in range(gt_boxes.shape[0]):
            box_cls = int(gt_boxes[i, 4])
            top = gt_boxes[i, :2]
            bottom = top + gt_boxes[i, 2: 4]
            cv2.rectangle(img_, tuple(np.round(top).astype(int).tolist()),
                          tuple(np.round(bottom).astype(int).tolist()), (255, 0, 0), 2)
            cv2.putText(img_, self.cls_names[box_cls][:3], (int(top[0]), int(top[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0, 0, 255), 1, 1)


def create_coco_loader(coco_dataset: Dataset, batch_size=2, shuffle=False):
    coco_loader = DataLoader(coco_dataset, batch_size, shuffle, num_workers=1, collate_fn=COCODataset.collate)
    return coco_loader


def convert_to_tensor(batch_data: Dict, to_gpu=False) -> Dict:
    """
    :param batch_data: output of COCODataset.collate() - {'images': (B, H, W, 3), 'gt_boxes': [(N0, 5), ...]}
    :param to_gpu: if True, values in batch_data will be moved to gpu
    :return: {'images': (B, 3, H, W), 'gt_boxes': [(N0, 5), ...]}
    """
    batch_data['images'] = torch.from_numpy(batch_data['images']).float().permute(0, 3, 1, 2).contiguous()
    batch_data['gt_boxes'] = [torch.from_numpy(boxes).float() for boxes in batch_data['gt_boxes']]
    if to_gpu:
        batch_data['images'] = batch_data['images'].cuda()
        batch_data['gt_boxes'] = [boxes.cuda() for boxes in batch_data['gt_boxes']]
    return batch_data
