# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmseg.models import builder
# from mmseg.core import add_prefix

# @builder.build_loss()
# class CrossEntropyDiceLoss(nn.Module):
#     def __init__(self, use_sigmoid=True, use_voting=False, ignore_index=-1, reduction='mean', loss_weight=1.0):
#         super(CrossEntropyDiceLoss, self).__init__()
#         self.use_sigmoid = use_sigmoid
#         self.use_voting = use_voting
#         self.ignore_index = ignore_index
#         self.reduction = reduction
#         self.loss_weight = loss_weight
#         self.cross_entropy = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    
#     def forward(self, cls_score, label):
#         if self.use_sigmoid:
#             cls_score = cls_score.sigmoid()
#         if self.use_voting:
#             cls_score = F.avg_pool2d(cls_score, kernel_size=3, stride=1, padding=1)
#         loss_ce = self.cross_entropy(cls_score, label)
        
#         # Calculate Dice Loss
#         prob = F.softmax(cls_score, dim=1)
#         prob = prob[:, 1, ...]  # Assuming binary segmentation
#         label = label == 1  # Assuming binary segmentation
#         smooth = 1e-5
#         intersection = (prob * label).sum(dim=(2, 3))
#         union = prob.sum(dim=(2, 3)) + label.sum(dim=(2, 3)) + smooth
#         dice = 1 - (2 * intersection + smooth) / union
#         loss_dice = dice.mean()

#         loss = self.loss_weight * loss_ce + (1 - self.loss_weight) * loss_dice

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'none':
#             return loss
#         else:
#             raise ValueError(f'Unsupported reduction: {self.reduction}')
