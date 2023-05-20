from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
sys.path.append('../')

import argparse
import cv2
import torch
from glob import glob
import torchvision.utils as vutils
from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.siamfdb_tracker import SiamFDBTracker
from pysot.utils.model_load import load_pretrain
from CAM.GroupCAM import GroupCAM
from toolkit.datasets import DatasetFactory
import numpy as np
import torch.nn as nn
from pysot.utils.bbox import get_axis_aligned_bbox
torch.set_num_threads(1)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
parser = argparse.ArgumentParser(description='SiamFDB demo')
parser.add_argument('--video', default='', type=str, help='eval one special video')
parser.add_argument('--dataset_dir', type=str, help='dataset root directory')
parser.add_argument('--dataset', type=str, help='dataset name')
parser.add_argument('--snapshot', type=str, default='B64W/checkpoint_e11.pth', help='model name')
parser.add_argument('--format', default='pdf', type=str, help='png, pdf, jpg')
parser.add_argument('--save_dir', default='./heatmap', type=str, help='Save path')
parser.add_argument('--config', type=str, default='./experiments/SiamFDB_r50_got10k/config.yaml', help='config file')

args = parser.parse_args()

def show_cam(img, mask, idx, title=None, title2=None):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        img (Tensor): shape (1, 3, H, W)
        mask (Tensor): shape (1, 1, H, W)
    Return:
        heatmap (Tensor): shape (3, H, W)
        cam (Tensor): synthesized GradCAM cam of the same shape with heatmap.
        :param title:
    """
    mask = (mask - mask.min()).div(mask.max() - mask.min()).data
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze().float()), cv2.COLORMAP_JET)  # [H, W, 3]
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # cam = heatmap + img.cpu()
    cam = 1 * (1 - mask ** 0.8) * img + (mask ** 0.8) * heatmap
    if title is not None:
        vutils.save_image(cam, title)
    if title2 is not None:
        vutils.save_image(heatmap, title2)

    return cam


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = sorted(glob(os.path.join(video_name, '*.jp*')))
        for img in images:
            frame = cv2.imread(img)
            yield frame

class Model(nn.Module):
    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model
        self.relu = nn.ReLU()
    def template(self, z):
        self.model.template(z)
    def track(self, x):
        data = self.model.track(x)
        cls = data['cls']
        cls = self.relu(cls)
        return {
            'cls': cls,
            'loc': data['loc'],
        }

def main():
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')
    # create model
    model = ModelBuilder()
    # load model
    model = load_pretrain(model, args.snapshot).eval().to(device)
    model = Model(model)
    print(model)
    # build tracker
    tracker = SiamFDBTracker(model, cfg.TRACK)
    CAM = GroupCAM(tracker, target_layer="Model.relu")
    if args.dataset == "GOT-10k":
        dataset = "GOT10k"
    else:
        dataset = args.dataset
    params = getattr(cfg.HP_SEARCH, dataset)
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    if 'GOT' in args.dataset:
        dataset_root = os.path.join(args.dataset_dir, dataset, "test")
    else:
        dataset_root = os.path.join(args.dataset_dir, args.dataset)
    dataset = DatasetFactory.create_dataset(name=args.dataset, dataset_root=dataset_root, load_img=True)
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            if video.name != args.video:
                continue
        for idx, (img, gt_bbox) in enumerate(video):
            if len(gt_bbox) == 4:
                gt_bbox = [gt_bbox[0], gt_bbox[1], gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                           gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1, gt_bbox[0] + gt_bbox[2] - 1,
                           gt_bbox[1]]
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
                CAM.model.init(img, gt_bbox_)
            else:
                outputs, outimg = CAM(img, hp)
                if outputs is None:
                    continue
                fusionpath = os.path.join(args.save_dir,"heatmap"， args.dataset,video.name)
                if not os.path.exists(fusionpath):
                    os.makedirs(fusionpath)
                heatmappath = os.path.join(args.save_dir, "fusion"，args.dataset, video.name)
                if not os.path.exists(heatmappath):
                    os.makedirs(heatmappath)
                show_cam(outimg, outputs, idx, os.path.join(fusionpath, "{:06d}.{}".format(idx,args.format)),
                         os.path.join(heatmappath,  "{:06d}.{}".format(idx,args.format)))

if __name__ == '__main__':
    main()
