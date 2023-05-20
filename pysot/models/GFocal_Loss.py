import torch.nn as nn
import torch.nn.functional as F
import torch
from pysot.core.config import cfg
# 默认
class QualityFocalLoss(nn.Module):

    def __init__(self, beta=2.0):
        super(QualityFocalLoss, self).__init__()
        self.beta = beta

    def forward(self, pred, target):
        label = target.view(-1)
        pos = torch.nonzero(label > 0).squeeze(1)
        neg = torch.nonzero(label == 0).squeeze(1)

        pos_num = max(pos.numel(),1.0)
        neg_num = max(neg.numel(),1.0)

        mask = ~(label == -1)
        pred = pred.view(-1, 2)
        label1 = label.view(-1, 1)
        label2 = torch.zeros(label1.shape).to(label1.device)
        label2[neg] = 1.0
        labels = torch.cat([label2, label1], dim=1).to(pred.device)

        pred = pred[mask]
        labels = labels[mask]

        loss_pos = self.get_gfl_loss(pred[:,-1], labels[:, -1]).sum()/pos_num
        loss_neg = self.get_gfl_loss(pred[:, 0], labels[:,  0]).sum()/neg_num

        return loss_pos*0.5 + loss_neg *0.5 

    def get_gfl_loss(self, pred, label):
        scale_factor = (pred - label).abs().pow(self.beta)
        loss = F.binary_cross_entropy(pred, label, reduction='none') * scale_factor
        return loss


def bbox_overlaps(pred, target):
    if cfg.TRAIN.REG_MAX > 0:
        pred = pred * cfg.TRACK.STRIDE
        target = target * cfg.TRACK.STRIDE
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_aera = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_aera = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + \
                  torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + \
                  torch.min(pred_top, target_top)

    area_intersect = w_intersect * h_intersect
    area_union = target_aera + pred_aera - area_intersect
    eps = area_union.new_tensor([1e-5])
    area_union = torch.max(area_union, eps)
    iou = area_intersect / area_union
    return iou

class DistributionFocalLoss(nn.Module):
    def __init__(self, beta=0):
        super(DistributionFocalLoss, self).__init__()

    def forward(self, pred, target, weight=None):
        disl = target.long()
        disr = disl + 1
        wl = disr.float() - target
        wr = target - disl.float()
        losses = F.cross_entropy(pred, disl, reduction='none') * wl + F.cross_entropy(pred, disr, reduction='none') * wr
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()/weight.sum()/4.0
        else:
            assert losses.numel() != 0
            return losses.mean()
  

if __name__ == "__main__":
    pred = torch.rand(50, 50, 4)
    target = torch.rand(50, 4)
    loss = QualityFocalLoss()
    print(pred.shape)
    print(loss(pred, target))


    # a = torch.Tensor([1, 1]).long() # 2
    # print(a.shape)
    # b = torch.Tensor([[0.8, 0.1, 0.1], [0.9, 0.05, 0.05]]) # 2,3
    # print(b.shape)
    # print(F.cross_entropy(input=b, target=a))
