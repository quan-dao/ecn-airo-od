import torch
import numpy as np
from typing import Tuple, List, Union, Dict


def gaussian_radius(det_size, min_overlap=0.3):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)


def draw_gaussian(mean: Union[np.ndarray, torch.Tensor], radius: float, class_heat_map_: torch.Tensor) -> None:
    """
    Generate a 2D gaussian
    :param mean: (2) - mean_x, mean_y
    :param radius: (i.e. Gaussian's std deviation)
    :param class_heat_map_: (H, W) - "_" indicates this argument is mutated by this function
    """
    assert len(class_heat_map_.shape) == 2, f"{class_heat_map_.shape}"
    imsize = class_heat_map_.shape
    xy = torch.zeros(imsize[0] * imsize[1], 2)  # (N, 2) TODO
    if not isinstance(mean, torch.Tensor):
        mean = torch.from_numpy(mean).float()  # (2)
    prob = torch.zeros(xy.shape[0])  # (N) TODO
    # TODO: splash prob onto class_heat_map_


def generate_regression_target(gt_boxes: Union[torch.Tensor, np.ndarray], downsample_factor: int,
                               output_size: Union[Tuple, List]) -> Tuple:
    """
    Generate CenterNet regression target (width_height map & subpixel_offset map)
    :param gt_boxes: (N, 5) - top_x, top_y, w, h, class
    :param downsample_factor: input_image_size / output_size (> 1)
    :param output_size: (height, width) of CenterNet's output
    :return: target_wh & target_offset - (N_channels, H_out, W_out)
    """
    assert downsample_factor >= 1
    if isinstance(gt_boxes, np.ndarray):
        gt_boxes = torch.from_numpy(gt_boxes)
    target_wh = torch.zeros(2, *output_size, dtype=torch.float)
    target_off = torch.zeros(2, *output_size, dtype=torch.float)
    n_boxes = gt_boxes.shape[0]
    centers = torch.zeros(n_boxes, 2)  # (N, 2) - @ the scale of input TODO
    centers = torch.zeros(n_boxes, 2)  # (N, 2) - @ the scale of output TODO
    centers_int = centers  # (N, 2) TODO
    centers_ = centers_int.long()  # (N, 2) - use this to index into target_wh & target_off

    # compute regression target
    offset = torch.zeros(n_boxes, 2)  # (N, 2) TODO
    wh = torch.zeros(n_boxes, 2)  # (N, 2) - @ the scale of output TODO

    # TODO: step 6
    return target_wh, target_off


class CenterNetTargetAssigner(object):
    def __init__(self, input_size: Tuple[int], output_stride: int, num_classes: int, centerness_min_iou: float = 0.3):
        """
        :param input_size: (height, width) of input images
        :param output_stride: input_image_size / output_size (> 1)
        :param num_classes:
        :param centerness_min_iou:
        """
        assert output_stride >= 1, f"{output_stride} not >= 1"
        self.input_size = input_size
        self.output_stride = output_stride
        self.output_size = tuple([int(float(s) / float(output_stride)) for s in self.input_size])
        self.n_cls = num_classes
        self.centerness_min_iou = centerness_min_iou

    def __call__(self, batched_gt_boxes: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        :param batched_gt_boxes: [(N0, 5), (N1, 5), ...] 5 == top_x, top_y, w, h, class | coord & size @ scale of input
        :return: {'target_heat_map': (B, n_cls, H_out, W_out),
                'target_wh': (B, 2, H_out, W_out), 'target_offset': (B, 2, H_out, W_out)}
        """
        batch_size = len(batched_gt_boxes)
        target_heat_map = torch.zeros(batch_size, self.n_cls, *self.output_size)
        target_wh = []
        target_offset = []
        for batch_idx in range(batch_size):
            cur_gt_boxes = batched_gt_boxes[batch_idx]  # (N, 5)
            # generate heat map
            for box_idx in range(cur_gt_boxes.shape[0]):
                box = cur_gt_boxes[box_idx]
                radius = gaussian_radius((0, 0))  # TODO: provide input for gaussian_radius
                center = torch.zeros(2)  # TODO: compute box's center
                draw_gaussian(center, radius, target_heat_map[batch_idx, box[-1].long()])

            # generate regression target
            gt_wh, gt_offset = generate_regression_target(None, None, None)  # TODO: provide input
            target_wh.append(gt_wh.unsqueeze(0))
            target_offset.append(gt_offset.unsqueeze(0))
        target_wh = torch.cat(target_wh, dim=0)
        target_offset = torch.cat(target_offset, dim=0)
        return {
            'target_cls': target_heat_map,
            'target_wh': target_wh,
            'target_offset': target_offset
        }
