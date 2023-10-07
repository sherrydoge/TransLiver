import torch
import torch.nn as nn


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if penalty == 'l2':
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets, is_coe=False):
        # get batch size
        N = targets.size()[0]
        smooth = 1e-5
        # flatten
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        # clip
        input_flat = torch.clamp(input_flat, 0, 1)
        targets_flat = torch.clamp(targets_flat, 0, 1)
        # calculate intersection
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)
        # calculate loss
        if is_coe:
            loss = N_dice_eff.sum() / N
        else:
            loss = 1 - N_dice_eff.sum() / N
        return loss
