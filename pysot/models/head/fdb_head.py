import torch
from torch import nn
import math
from pysot.core.config import cfg
import torch.nn.functional as F

class FDBHead(torch.nn.Module):
    def __init__(self, cfg, in_channels, add_mean=True):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FDBHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.TRAIN.NUM_CLASSES
        self.reg_max = cfg.TRAIN.REG_MAX
        self.reg_topk = cfg.TRAIN.REG_TOPK
        self.total_dim = cfg.TRAIN.REG_TOPK
        self.add_mean = add_mean
        if add_mean:
            self.total_dim += 1
        cls_tower = []
        bbox_tower = []
        self.Scale = Scale(1.0)
        for i in range(cfg.TRAIN.NUM_CONVS):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())

            bbox_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1,padding=1))
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())


        self.reg_conf = nn.Sequential(
            nn.Conv2d(4*self.total_dim,64,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,1,1),
            nn.Sigmoid()
        )
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3,stride=1,padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4 * (self.reg_max + 1), kernel_size=3, stride=1, padding=1)
        self.cls_sigmoid = nn.Sigmoid()
        for l in self.reg_conf:
            if isinstance(l, nn.Conv2d):
                torch.nn.init.normal_(l.weight, std=0.01)
                torch.nn.init.constant_(l.bias, 0)
        # initialization
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        # initialize the bias for focal loss
        prior_prob = cfg.TRAIN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        torch.nn.init.normal_(self.bbox_pred.weight, std=0.01)
        torch.nn.init.constant_(self.bbox_pred.bias, 0)


    def forward(self, xc, xr):
        cls_tower = self.cls_tower(xc)
        logits = self.cls_sigmoid(self.cls_logits(cls_tower))
        bbox_reg = self.bbox_pred(self.bbox_tower(xr))
        if cfg.TRAIN.REG_MAX == 0:
            bbox_reg = torch.exp(bbox_reg)
        elif cfg.TRAIN.IOUGUIDED:
            N,C,H,W = bbox_reg.size()
            prob = F.softmax(bbox_reg.reshape(N,4,self.reg_max+1,H,W),dim=2)
            prob_topk,_ = prob.topk(self.reg_topk, dim=2)
            if self.add_mean:
                stat = torch.cat([prob_topk,prob_topk.mean(dim=2, keepdim=True)],dim=2).to(bbox_reg.device)
            else:
                stat = prob_topk
            qs = self.reg_conf(stat.reshape(N,-1,H,W))
            
            if cfg.TRAIN.NUM_CLASSES == 2:
                background = logits[:,0,:,:].reshape(N,-1,H,W)
                foreground = logits[:,-1,:,:].reshape(N,-1,H,W) * qs
                logits = torch.cat([background,foreground],dim=1)
            else:
                logits = logits * qs

        return logits, bbox_reg


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


