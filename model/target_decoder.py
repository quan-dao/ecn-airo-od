import torch
import torch.nn.functional as F
from typing import Dict, List


def decode_centernet_target(pred_dict: Dict[str, torch.Tensor], output_stride: int, cls_is_activated: bool,
                            max_num_obj: int = 100) -> torch.Tensor:
    """
    :param pred_dict: {
            'cls': (B, n_cls, H_out, W_out),
            'wh': (B, 2, H_out, W_out),
            'reg': (B, 2, H_out, W_out)
            }
    :param output_stride: = input size / output size
    :param cls_is_activated: if False, cls has not been acitvated by sigmoid
    :param max_num_obj:
    :return: (B, max_num_obj, 6) - 6:= top_x, top_y, w, h, cls_idx, confidence
    """
    assert output_stride >= 1, f"{output_stride} not >= 1"
    batch_size, _, H_out, W_out = pred_dict['cls'].shape
    # id peak in cls
    peak = pred_dict['cls']  # TODO: step 1
    mask_peak = pred_dict['cls'] == peak  # (B, n_cls, H_out, W_out)
    heatmap = pred_dict['cls'] * mask_peak.float()  # (B, n_cls, H_out, W_out) - non-peak are zero out
    # TODO: check the lab subject
    out = torch.zeros(batch_size, max_num_obj, 6)
    return out
