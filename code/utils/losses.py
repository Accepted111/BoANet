import torch
from torch.nn import functional as F


def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss


def weighted_cross_entropy_loss(pred, mask):
    mask = mask.float()
    weight = 1 + 5 * torch.abs(F.avg_pool3d(mask, kernel_size=21, stride=1, padding=10) - mask)
    wce = F.binary_cross_entropy_with_logits(pred, mask)
    wce = (weight*wce).sum(dim=(1, 2, 3)) / weight.sum(dim=(1, 2, 3))
    return wce.mean()

