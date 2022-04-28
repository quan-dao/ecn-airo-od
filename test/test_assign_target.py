import unittest
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from dataset.coco_dataset import COCODataset, create_coco_loader
from model.assign_target import draw_gaussian, gaussian_radius, CenterNetTargetAssigner


class MyTestCase(unittest.TestCase):
    def test_draw_gaussian(self):
        coco_dataset = COCODataset()
        data_dict = coco_dataset[15]
        img = data_dict['image']
        annos = data_dict['gt_boxes']
        unique_labels, anno_ids_to_label_ids = np.unique(annos[:, 4].astype(int), return_inverse=True)
        num_labels = unique_labels.shape[0]
        # generate target heat map
        print(f"annos {annos.shape}:\n{annos})")
        heat_map = torch.zeros(num_labels, *coco_dataset.input_size)
        for gt_idx in range(annos.shape[0]):
            radius = gaussian_radius(annos[gt_idx, [3, 2]])
            mean = annos[gt_idx, :2] + annos[gt_idx, 2: 4] / 2.0
            draw_gaussian(mean, radius, heat_map[anno_ids_to_label_ids[gt_idx]])

        # vis
        fig, axe = plt.subplots(1, num_labels)
        for j, cls_id in enumerate(unique_labels.tolist()):
            img_ = np.copy(img)
            for gt_idx in range(annos.shape[0]):
                if int(annos[gt_idx, 4]) != cls_id:
                    continue
                top = annos[gt_idx, :2]
                bottom = top + annos[gt_idx, 2: 4]
                cv2.rectangle(img_, tuple(np.round(top).astype(int).tolist()),
                              tuple(np.round(bottom).astype(int).tolist()), (255, 0, 0), 2)
            axe[j].imshow(img_[..., ::-1])
            axe[j].imshow(heat_map[j], alpha=0.35)
            axe[j].set_title(f"target heat map of cls {coco_dataset.cls_names[cls_id]}")
        plt.show()

    def test_target_assigner(self):
        coco_dataset = COCODataset()
        batch_size = 2
        coco_loader = create_coco_loader(coco_dataset, batch_size)
        coco_loader_iter = iter(coco_loader)
        batch_dict = None
        for dummy in range(3):
            batch_dict = next(coco_loader_iter)
        target_assigner = CenterNetTargetAssigner(coco_dataset.input_size, output_stride=1, num_classes=80)
        batch_gt_boxes = [torch.from_numpy(boxes).float() for boxes in batch_dict['gt_boxes']]
        target_dict = target_assigner(batch_gt_boxes)
        target_heat_map = target_dict['target_cls']  # (B, 80, H, W)
        images = batch_dict['images']  # (B, H, W, 3)
        figs, axes = [], []
        for bidx in range(batch_size):
            img = images[bidx]
            heat_map = target_heat_map[bidx]  # (80, H, W)
            annos = batch_dict['gt_boxes'][bidx]
            unique_labels, anno_ids_to_label_ids = np.unique(annos[:, 4].astype(int), return_inverse=True)
            num_labels = unique_labels.shape[0]
            print(f"num_labels: {num_labels}")
            fig, axe = plt.subplots(1, num_labels)
            for j, cls_idx in enumerate(unique_labels.tolist()):
                img_ = np.copy(img)
                mask_cls = annos[:, 4].astype(int) == cls_idx
                coco_dataset.draw_annotation(img_, annos[mask_cls])
                if num_labels > 1:
                    axe[j].imshow(img_[..., ::-1])
                    axe[j].imshow(heat_map[cls_idx], alpha=0.5)
                    axe[j].set_title(f"target heat map of cls {coco_dataset.cls_names[cls_idx]}")
                else:
                    axe.imshow(img_[..., ::-1])
                    axe.imshow(heat_map[cls_idx], alpha=0.5)
                    axe.set_title(f"target heat map of cls {coco_dataset.cls_names[cls_idx]}")

            # ---
            figs.append(fig)
            axes.append(axe)

        plt.show()





