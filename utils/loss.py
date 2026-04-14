import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logpt = -F.cross_entropy(input, target, weight=self.weight,
                                 ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss[target != self.ignore_index].mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, input, target):
        num_classes = input.shape[1]
        input = F.softmax(input, dim=1)
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        # 忽略 ignore_index 区域
        valid_mask = (target != self.ignore_index).unsqueeze(1)
        input = input * valid_mask
        target_one_hot = target_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(input * target_one_hot, dims)
        union = torch.sum(input + target_one_hot, dims)

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-5, ignore_index=255):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, input, target):
        num_classes = input.shape[1]
        input = F.softmax(input, dim=1)
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()

        mask = (target != self.ignore_index).unsqueeze(1)
        input = input * mask
        target_one_hot = target_one_hot * mask

        dims = (0, 2, 3)
        TP = torch.sum(input * target_one_hot, dims)
        FP = torch.sum(input * (1 - target_one_hot), dims)
        FN = torch.sum((1 - input) * target_one_hot, dims)

        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        return 1 - tversky.mean()


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_freq=None, ignore_index=255):
        super(ClassBalancedFocalLoss, self).__init__()
        self.gamma = gamma
        self.class_freq = class_freq
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if self.class_freq is not None:
            weight = torch.tensor(self.class_freq, dtype=torch.float32, device=input.device)
        else:
            weight = None
        logpt = -F.cross_entropy(input, target, weight=weight,
                                 ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss[target != self.ignore_index].mean()


class HybridLossV2(nn.Module):
    def __init__(self, losses=["focal", "dice"], weights=[0.5, 0.5], loss_args=None):
        super(HybridLossV2, self).__init__()
        self.losses = []
        if loss_args is None:
            loss_args = [{} for _ in losses]
        for name, args in zip(losses, loss_args):
            if name == "focal":
                self.losses.append(FocalLoss(**args))
            elif name == "dice":
                self.losses.append(DiceLoss(**args))
            elif name == "tversky":
                self.losses.append(TverskyLoss(**args))
            elif name == "cbfocal":
                self.losses.append(ClassBalancedFocalLoss(**args))
            else:
                raise ValueError(f"Unsupported loss: {name}")
        self.weights = weights

    def forward(self, input, target):
        total_loss = 0
        for loss_fn, w in zip(self.losses, self.weights):
            total_loss += w * loss_fn(input, target)
        return total_loss
