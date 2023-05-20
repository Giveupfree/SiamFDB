"""
This file contains specific functions for computing losses of SiamFDB
file
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from pysot.models.GFocal_Loss import QualityFocalLoss,DistributionFocalLoss
import math
from pysot.core.config import cfg

INF = 100000000


def get_gfl_loss(pred, label, select, beta=2):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = pred[select]
    label = label[select]
    scale_factor = (pred - label).abs().pow(beta)
    loss = F.binary_cross_entropy(pred, label, reduction='none') * scale_factor
    # loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none') * scale_factor
    return loss.mean()


# 1.统计每次正负样本数目
def select_qfl(pred, label, pos, neg):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    if cfg.TRAIN.POINT == 'ellipse':
        pass
    else:
        label1 = label.view(-1, 1)
        label2 = torch.zeros(label1.shape).to(label1.device)
        label2[neg] = 1.0
        labels = torch.cat([label2, label1], dim=1).to(pred.device)
    loss_pos = get_gfl_loss(pred, labels, pos)
    loss_neg = get_gfl_loss(pred, labels, neg)
    return (loss_pos * 0.5 + loss_neg * 0.5)


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


def get_dfl_loss(pred, label, weight):
    disl = label.long()
    disr = disl + 1
    wl = disr.float() - label
    wr = label - disl.float()
    losses = F.cross_entropy(pred, disl, reduction='none') * wl \
         + F.cross_entropy(pred, disr, reduction='none') * wr
    if weight is not None and weight.sum() > 0:
        return (losses * weight).sum()
    else:
        assert losses.numel() != 0
        return losses.mean()


class IOULoss(nn.Module):
    def __init__(self, name="iou", gama=0.5):
        super(IOULoss, self).__init__()
        self.name = name
        self.gama = gama

    def forward(self, pred, target, weight=None):
        if cfg.TRAIN.REG_MAX>0:
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

        enclose_x1y1 = torch.max(pred[:, :2], target[:, :2])
        enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
        enclose_wh = (enclose_x2y2 + enclose_x1y1).clamp(min=0)
        enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]

        #()
        predcenter_x = (pred_right - pred_left) / 2.0
        predcenter_y = (pred_top - pred_bottom) / 2.0
        targetcenter_x = (target_right - target_left) / 2.0
        targetcenter_y = (target_top - target_bottom) / 2.0

        inter_diag = (predcenter_x - targetcenter_x) ** 2 + (predcenter_y - targetcenter_y) ** 2

        outer_diag = enclose_wh[:, 0] ** 2 + enclose_wh[:, 1] ** 2

        if self.name == "logiou":
            losses = - torch.log(iou)
        elif self.name == "giou":
            gious = iou - (enclose_area - area_union) / enclose_area
            losses = 1 - gious
        elif "diou" in self.name:
            losses = 1 - iou + inter_diag / outer_diag
            if "log" in self.name:
                losses = - torch.log(1 - losses/2)
        elif "ciou" in self.name:
            w2 = target_left + target_right
            h2 = target_top + target_bottom
            w1 = pred_left + pred_right
            h1 = pred_top + pred_bottom
            v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
            S = 1 - iou
            k = torch.max(S + v, eps)
            alpha = v / k
            losses = 1 - iou + inter_diag / outer_diag + alpha * v
            if "log" in self.name:
                losses = - torch.log(1 - losses/3)
        elif "eiou" in self.name:
            # 翻译结果
            lw = (target_left + target_right - (pred_left + pred_right)) ** 2.0
            lh = (target_top + target_bottom - (pred_top + pred_bottom)) ** 2.0
            outer_diag = torch.max(outer_diag, eps)
            enclose_w = torch.max(enclose_wh[:, 0], eps)
            enclose_h = torch.max(enclose_wh[:, 1], eps)
            losses = 1 - iou + inter_diag / outer_diag + lw/enclose_w**2 + lh/enclose_h**2
            if "f" in self.name or "F" in self.name:
                losses = losses * (iou ** self.gama)
            if "log" in self.name:
                losses = - torch.log(1 - losses/4)
        else:
            losses = 1 - iou

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()/ weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class SiamFDBLossComputation(object):
    """
    This class computes the SiamFDB losses.
    """

    def __init__(self, cfg):
        self.box_reg_loss_func = IOULoss('logiou')
        self.qfl = QualityFocalLoss()
        self.dfl = DistributionFocalLoss()
        self.cfg = cfg
        self.reg_max = cfg.TRAIN.REG_MAX
        if self.reg_max > 0:
            self.distribution_project = Project(self.reg_max)
            self.dt = 0.01

    def prepare_targets(self, locations, labels, gt_bbox):
        bboxes = gt_bbox
        labels = labels.view(self.cfg.TRAIN.OUTPUT_SIZE ** 2, -1)

        xs, ys = locations[:, 0], locations[:, 1]

        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], -1)
        if cfg.TRAIN.POINT == 'ellipse':
                # bboxes[:, 0]
            tcx, tcy, tw, th = (bboxes[:, 0] + bboxes[:, 2]) * 0.5, (bboxes[:, 1] + bboxes[:, 3]) * 0.5, \
                                   bboxes[:,2] - bboxes[:,0], bboxes[:,3] - bboxes[:,1]
            pos = torch.where(torch.square(tcx - xs) / torch.square(tw / 4) +
                                  torch.square(tcy - ys) / torch.square(th / 4) < 1)
            neg = torch.where(torch.square(tcx - xs) / torch.square(tw / 2) +
                                  torch.square(tcy - ys) / torch.square(th / 2) > 1)

            labels[pos] =  1
            labels[neg] = -1
        elif cfg.TRAIN.POINT == 'radius':
            radius_areas_point = torch.zeros_like(labels)
            radius = 1.0
            R = radius * cfg.TRACK.STRIDE
            tcx, tcy, tw, th = (bboxes[:, 0] + bboxes[:, 2]) * 0.5, (bboxes[:, 1] + bboxes[:, 3]) * 0.5, \
                               bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]


            pos = torch.where(torch.square(tcx - xs)/torch.square(R) + torch.square(tcy - ys) / torch.square(R) < 1)
            radius_areas_point[pos] = 1

            s1 = reg_targets_per_im[:, :, 0] > 0
            s2 = reg_targets_per_im[:, :, 2] > 0
            s3 = reg_targets_per_im[:, :, 1] > 0
            s4 = reg_targets_per_im[:, :, 3] > 0

            is_in_boxes = s1 * s2 * s3 * s4
            pos = np.where(is_in_boxes.cpu() == 1)
            labels[pos] = 1
            labels = radius_areas_point * labels
        else:
            s1 = reg_targets_per_im[:, :, 0] > 0
            s2 = reg_targets_per_im[:, :, 2] > 0
            s3 = reg_targets_per_im[:, :, 1] > 0
            s4 = reg_targets_per_im[:, :, 3] > 0
            is_in_boxes = s1 * s2 * s3 * s4
            pos = np.where(is_in_boxes.cpu() == 1)
            labels[pos] = -1
            s1 = reg_targets_per_im[:, :, 0] > 0.5 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
            s2 = reg_targets_per_im[:, :, 2] > 0.5 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
            s3 = reg_targets_per_im[:, :, 1] > 0.5 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
            s4 = reg_targets_per_im[:, :, 3] > 0.5 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
            # s1 = reg_targets_per_im[:, :, 0] > 0.5 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
            # s2 = reg_targets_per_im[:, :, 2] > 0.5 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
            # s3 = reg_targets_per_im[:, :, 1] > 0.5 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
            # s4 = reg_targets_per_im[:, :, 3] > 0.5 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
            is_in_boxes = s1 * s2 * s3 * s4
            pos = np.where(is_in_boxes.cpu() == 1)
            labels[pos] = 1

        return labels.permute(1, 0).contiguous(), reg_targets_per_im.permute(1, 0, 2).contiguous()

    def bbox2distrance(self, reg_targets):
        l = reg_targets[:, 0].clamp(min=0, max=self.reg_max - self.dt)
        t = reg_targets[:, 1].clamp(min=0, max=self.reg_max - self.dt)
        r = reg_targets[:, 2].clamp(min=0, max=self.reg_max - self.dt)
        b = reg_targets[:, 3].clamp(min=0, max=self.reg_max - self.dt)
        reg_targets_per_im = torch.stack([l, t, r, b], dim=-1)
        return reg_targets_per_im

    def compute_cls_targets(self, reg_targets, box_regression=None):
        if cfg.TRAIN.REG_MAX > 0:
            reg_targets = reg_targets * cfg.TRACK.STRIDE
            box_regression = box_regression * cfg.TRACK.STRIDE
        with torch.no_grad():
            if self.cfg.TRAIN.IOU and box_regression is not None:
                pred_left = box_regression[:, 0]
                pred_top = box_regression[:, 1]
                pred_right = box_regression[:, 2]
                pred_bottom = box_regression[:, 3]

                target_left = reg_targets[:, 0]
                target_top = reg_targets[:, 1]
                target_right = reg_targets[:, 2]
                target_bottom = reg_targets[:, 3]

                target_aera = (target_left + target_right) * \
                              (target_top + target_bottom)
                pred_aera = (pred_left + pred_right) * \
                            (pred_top + pred_bottom)

                w_intersect = torch.min(pred_left, target_left).type_as(reg_targets) + \
                              torch.min(pred_right, target_right).type_as(reg_targets)
                h_intersect = torch.min(pred_bottom, target_bottom).type_as(reg_targets) + \
                              torch.min(pred_top, target_top).type_as(reg_targets)

                area_intersect = w_intersect * h_intersect
                area_union = target_aera + pred_aera - area_intersect
                eps = area_union.new_tensor([1e-5])
                area_union = torch.max(area_union, eps)
                iou = area_intersect/area_union
                return iou
            else:
                left_right = reg_targets[:, [0, 2]]
                top_bottom = reg_targets[:, [1, 3]]
                centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                             (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
                return torch.sqrt(centerness)


    def __call__(self, locations, box_cls, box_regression, labels, reg_targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            dfl_loss (Tensor)
        """
        if self.reg_max > 0:
            locations = locations / cfg.TRACK.STRIDE
            reg_targets = reg_targets/cfg.TRACK.STRIDE

        label_cls, reg_targets= self.prepare_targets(locations, labels, reg_targets)
        labels_flatten = (label_cls.view(-1))
        reg_targets_flatten = (reg_targets.view(-1, 4))
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        # neg_inds = torch.nonzero(labels_flatten == 0).squeeze(1)

        iou_targets = torch.zeros(labels_flatten.shape).to(labels_flatten.device)
        if self.reg_max > 0:
            box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4 * (self.reg_max+1)))
            box_regression = box_regression_flatten[pos_inds]

            box_regression_flatten = self.distribution_project(box_regression).to(labels_flatten.device)
            pred_Integrate = box_regression.reshape(-1, self.reg_max + 1)

        else:
            box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))
            box_regression_flatten = box_regression_flatten[pos_inds]
            # print(box_regression_flatten.shape)
        reg_targets_flatten = reg_targets_flatten[pos_inds]
            # print(reg_targets_flatten.shape)
        labels_float = labels_flatten.to(reg_targets_flatten.dtype)

        if pos_inds.numel() > 0:
            iou_targets[pos_inds] = self.compute_cls_targets(reg_targets_flatten, box_regression_flatten)
            labels_float = labels_float * iou_targets
            weight = box_cls.reshape(-1, 2).detach()[:, 1][pos_inds]
                # print(labels_flatten.shape)
            reg_loss = self.box_reg_loss_func(
                    box_regression_flatten,
                    reg_targets_flatten,
                    weight
                )
            if self.reg_max > 0:
                loss_dfl = self.dfl(pred_Integrate, self.bbox2distrance(reg_targets_flatten).reshape(-1), weight[:, None].expand(-1,4).reshape(-1))
        else:
            reg_loss = box_regression_flatten.sum()
            labels_float = torch.zeros_like(labels_float).to(labels_float.device)
            if self.reg_max > 0:
                loss_dfl = pred_Integrate.sum()
        cls_loss = self.qfl(box_cls, labels_float)
        # cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)
        if self.reg_max > 0:
            return cls_loss, reg_loss, loss_dfl
        else:
            return cls_loss, reg_loss

def make_siamfdb_loss_evaluator(cfg):
    loss_evaluator = SiamFDBLossComputation(cfg)
    return loss_evaluator


def compute_centerness_targets(reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)



class Project(nn.Module):
    """
    A fixed project layer for distribution
    """
    def __init__(self, reg_max=16):
        super(Project, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))
#

    def forward(self, x):
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


if __name__ == "__main__":
    c = torch.rand(5, 5, 128, 4 * 17).cuda()
    dis = Project()
    # pos_inds = torch.nonzero(c > 0).squeeze(1)
    cls_loss = dis(c)
    print(cls_loss.shape)
    c = torch.rand(128, 2).cuda()
    a = torch.rand(128).cuda()
    b = torch.zeros(128)
    # pos_inds = torch.nonzero(c > 0).squeeze(1)
    cls_loss = select_qfl(c, a)
    print(cls_loss)

    # d = a[pos_inds]
    # m = compute_centerness_targets(d)
    # b[pos_inds] = m
    # h = b[pos_inds]
    # n = c.data.ne(0).nonzero()
    # n = select_qfl(a,c)

