# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.inception

from pysot.core.config import cfg
from pysot.models.loss_fdb import make_siamfdb_loss_evaluator,Project
from pysot.models.backbone import get_backbone
from pysot.models.head.fdb_head import FDBHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise
from pysot.models.neck.neck import AdjustLayer


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)


        # build car head
        if cfg.TRAIN.REG_MAX > 16:
            self.fdb_head = self.car_head = FDBHead(cfg, 256)
        else:
            self.car_head = FDBHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamfdb_loss_evaluator(cfg)
        self.reg_max = cfg.TRAIN.REG_MAX

        self.neckCT = AdjustLayer(1024, 256)
        self.neckCS = AdjustLayer(1024, 256)
        self.neckRT = AdjustLayer(1024, 256)
        self.neckRS = AdjustLayer(1024, 256)

        self.reg_max = cfg.TRAIN.REG_MAX
        if self.reg_max > 0:
            self.distribution_project = Project(self.reg_max)

    def template(self, z):
        zf = self.backbone(z)
        zfc = self.neckCT(zf)
        zfr = self.neckRT(zf)

        self.zfc = zfc
        self.zfr = zfr

    def track(self, x):
        xf = self.backbone(x)
        xfc = self.neckCS(xf)
        xfr = self.neckRS(xf)

        fc = self.xcorr_depthwise(xfc, self.zfc)
        fr = self.xcorr_depthwise(xfr, self.zfr)
        if cfg.TRAIN.REG_MAX > 16:
            cls, loc = self.fdb_head(fc,fr)
        else:
            cls, loc = self.car_head(fc,fr)
        if self.reg_max > 0:
            b,c,h,w = cls.shape
            loc = loc.permute(0, 2, 3, 1).contiguous().view(-1, self.reg_max+1)
            loc = self.distribution_project(loc)
            loc = loc.view(b, w, h, 4).permute(0, 3, 1, 2).contiguous() * cfg.TRACK.STRIDE
        return {
                'cls': cls,
                'loc': loc,
               }

    def cls_aixs(self, cls):
        # b, a2, h, w = cls.size()
        # cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 1).contiguous()
        # cls = -F.log_softmax(cls, dim=3)
        # cls = cls.sigmoid()
        return cls

    def forward(self, data):
        """ only used in training"""
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()
        zf = self.backbone(template)
        xf = self.backbone(search)

        zfc = self.neckCT(zf)
        zfr = self.neckRT(zf)
        xfc = self.neckCS(xf)
        xfr = self.neckRS(xf)

        fc = self.xcorr_depthwise(xfc, zfc)
        fr = self.xcorr_depthwise(xfr, zfr)

        cls, loc = self.fdb_head(fc,fr)
        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.cls_aixs(cls)
        if self.reg_max > 0:
            cls_loss, loc_loss, dfl_loss = self.loss_evaluator(locations, cls, loc, label_cls, label_loc)
        else:
            cls_loss, loc_loss = self.loss_evaluator(locations, cls, loc, label_cls, label_loc)
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'], outputs['loc_loss']= cls_loss, loc_loss
        if self.reg_max > 0:
            outputs['total_loss'] +=  cfg.TRAIN.DFL_WEIGHT * dfl_loss
            outputs['d_loss'] =  dfl_loss
        return outputs
