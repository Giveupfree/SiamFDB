from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys
import shutil
sys.path.append('../')
from pysot.core.config import cfg
from pysot.tracker.siamfdb_tracker import SiamFDBTracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder
from toolkit.utils.region import vot_overlap, vot_float2str

from toolkit.datasets import DatasetFactory

torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

class Tracker:
    def __init__(self,config, dataset, snapshot, vis=False):
        cfg.merge_from_file(config)
        params = getattr(cfg.HP_SEARCH, dataset)
        self.hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}
        model = ModelBuilder()
        # load model
        model = load_pretrain(model, snapshot).cuda().eval()
        # build tracker
        self.tracker = SiamFDBTracker(model, cfg.TRACK)
        self.vis = vis
        self.dataset = dataset

    def run_vot(self):
        # load config
        if self.dataset is "VOT2020":
            import tools.vot2020 as vot
            def _convert_anno_to_list(vot_anno):
                vot_anno = [vot_anno[0], vot_anno[1], vot_anno[2], vot_anno[3]]
                return vot_anno

            def _convert_image_path(image_path):
                return image_path

            handle = vot.VOT("rectangle")
            vot_anno = handle.region()
            gt_bbox_ = _convert_anno_to_list(vot_anno)
        # elif 'VOT' in self.dataset:
        #     import tools.vot as vot
        #     def _convert_anno_to_list(vot_anno):
        #         vot_anno = [vot_anno[0][0][0], vot_anno[0][0][1], vot_anno[0][1][0], vot_anno[0][1][1],
        #                     vot_anno[0][2][0], vot_anno[0][2][1], vot_anno[0][3][0], vot_anno[0][3][1]]
        #         return vot_anno
        #
        #     def _convert_image_path(image_path):
        #         image_path_new = image_path[20:- 2]
        #         return "".join(image_path_new)
        #     handle = vot.VOT("polygon")
        #     vot_anno_polygon = handle.region()
        #     vot_anno_polygon = _convert_anno_to_list(vot_anno_polygon)
        #     cx, cy, w, h = get_axis_aligned_bbox(np.array(vot_anno_polygon))
        #     gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
        # else:
        #     raise ValueError
        elif 'VOT' in self.dataset:
            import tools.vot as vot
            def _convert_image_path(image_path):
                image_path_new = image_path[20:- 2]
                return "".join(image_path_new)
            handle = vot.VOT("rectangle")
            selection = handle.region()
            lx, ly, w, h = selection.x, selection.y, selection.width, selection.height
            gt_bbox_ = [lx, ly, w, h]
        else:
            raise ValueError

        """Run tracker on VOT."""

        image_path = handle.frame()
        if not image_path:
            return
        image_path = _convert_image_path(image_path)
        image = cv2.imread(image_path)
        self.tracker.init(image, gt_bbox_)

        idx = 0
        # Track
        while True:
            image_path = handle.frame()
            if not image_path:
                break
            image_path = _convert_image_path(image_path)

            image = cv2.imread(image_path)
            out = self.tracker.track(image, self.hp)
            state = out['bbox']
            pred = vot.Rectangle(state[0], state[1], state[2], state[3])
            idx += 1
            if self.dataset is "VOT2020":
                handle.report(pred, 1.0)
            else:
                handle.report(pred)


def run_vot2020(snapshot):
    tracker = Tracker('./experiments/SiamFDB_r50/config.yaml', 'VOT2020',snapshot)
    tracker.run_vot()


def run_vot(dataset,snapshot):
    tracker = Tracker('./experiments/SiamFDB_r50/config.yaml', dataset, snapshot)
    tracker.run_vot()

def run_vot2019():
    tracker = Tracker('./experiments/SiamFDB_r50/config.yaml', 'VOT2019', './snapshotAll/checkpoint_e14.pth')
    tracker.run_vot()

def main():
    parser = argparse.ArgumentParser(description='Run VOT.')
    parser.add_argument('dataset', type=str)
    parser.add_argument('snapshot', type=str)
    args = parser.parse_args()
    run_vot(args.dataset, args.snapshot)


if __name__ == '__main__':
    main()
