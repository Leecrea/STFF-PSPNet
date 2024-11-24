import torch
import torch.nn as nn
from mmseg.models.losses import DiceLoss
from mmseg.models.losses import CrossEntropyLoss

from ..builder import LOSSES
from mmseg.registry import MODELS

from .utils import weighted_loss

@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss

@LOSSES.register_module(name='MyLoss')
class MyLoss(nn.Module):
    def __init__(self, ce_weight=0.6, dice_weight=0.4, loss_name='loss_my'):
        super(MyLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self._loss_name = loss_name
        
    def forward(self, pred, target, **kwargs):
        ce_loss = self.ce_loss(pred, target, **kwargs)
        dice_loss = self.dice_loss(pred, target, **kwargs)
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return total_loss


    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name