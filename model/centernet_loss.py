import torch
import torch.nn as nn


def log(x: torch.Tensor):
    return torch.log(torch.clamp(x, min=1e-5))  # to avoid taking log  of negative numbers


def focal_loss(pred: torch.Tensor, target: torch.Tensor, pred_is_activated: bool, alpha=2., beta=4.) -> torch.Tensor:
    """
    :param pred: (B, C, H, W)
    :param target: (B, C, H, W)
    :param pred_is_activated: if False, pred is activated by sigmoid before computing loss
    :param alpha:
    :param beta:
    :return:
    """
    if not pred_is_activated:
        pred = torch.sigmoid(pred)
    loss = torch.rand(1)  # TODO
    return loss

