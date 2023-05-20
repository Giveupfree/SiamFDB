# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamcar_r50"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
# __C.TRAIN.POINT = 'ellipse'
__C.TRAIN.POINT = 'Normal'

__C.TRAIN.REG_MAX = 32
__C.TRAIN.REG_TOPK = 4
__C.TRAIN.IOU = True
__C.TRAIN.IOUGUIDED = True

__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logGOT'

__C.TRAIN.SNAPSHOT_DIR = './snapshotGOT'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 15

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 2.0

__C.TRAIN.DFL_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.NUM_CLASSES = 2

__C.TRAIN.NUM_CONVS = 4

__C.TRAIN.PRIOR_PROB = 0.01

__C.TRAIN.LOSS_ALPHA = 0.25

__C.TRAIN.LOSS_GAMMA = 2.0

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18
# __C.DATASET.SEARCH.SCALE = 0

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# for detail discussion
__C.DATASET.NEG = 0.0

__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'YOUTUBEBB', 'DET', 'COCO', "GOT", 'LaSOT')

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = './train_dataset/vid/VID/crop511'          # VID dataset path
__C.DATASET.VID.ANNO = './train_dataset/vid/VID/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = 100000  # repeat until reach NUM_USE

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = './train_dataset/yt_bb/YutubeBB/crop511'  # YOUTUBEBB dataset path
__C.DATASET.YOUTUBEBB.ANNO = './train_dataset/yt_bb/YutubeBB/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = './train_dataset/coco/COCO/crop511'         # COCO dataset path
__C.DATASET.COCO.ANNO = './train_dataset/coco/COCO/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = './train_dataset/det/DET/crop511'           # DET dataset path
__C.DATASET.DET.ANNO = './train_dataset/det/DET/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = -1

__C.DATASET.GOT = CN()
__C.DATASET.GOT.ROOT = './train_dataset/got10k/GOT10k/crop511'         # GOT dataset path
__C.DATASET.GOT.ANNO = './train_dataset/got10k/GOT10k/train.json'
# __C.DATASET.GOT.FRAME_RANGE = 100
# __C.DATASET.GOT.NUM_USE = 200000
__C.DATASET.GOT.FRAME_RANGE = 50
__C.DATASET.GOT.NUM_USE = 100000

__C.DATASET.LaSOT = CN()
__C.DATASET.LaSOT.ROOT = './train_dataset/lasot/LaSOT/crop511'         # LaSOT dataset path
__C.DATASET.LaSOT.ANNO = './train_dataset/lasot/LaSOT/train.json'
__C.DATASET.LaSOT.FRAME_RANGE = 100
__C.DATASET.LaSOT.NUM_USE = 100000

__C.DATASET.VIDEOS_PER_EPOCH = 600000 #600000
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = './pretrained_models/resnet50.model'
# __C.BACKBONE.PRETRAINED = './snapshot/checkpoint_e10.pth'
# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer1', 'layer2', 'layer3']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.CAR = CN()

# RPN type
__C.CAR.TYPE = 'MultiCAR'

__C.CAR.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamCARTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

__C.TRACK.INTERPOLATION = False

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

__C.TRACK.STRIDE = 8.0


__C.TRACK.SCORE_SIZE = 25

__C.TRACK.hanming = True

__C.TRACK.NUM_K = 2

__C.TRACK.NUM_N = 1

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44


# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

# __C.HP_SEARCH.OTB100 = [0.35, 0.2, 0.45]
__C.HP_SEARCH.OTB100 = [0.6037291666153346, 0.20484205699557875, 0.5824052956484521]

# __C.HP_SEARCH.GOT10k = [0.7, 0.06, 0.1]
# __C.HP_SEARCH.GOT10k = [0.2, 0.24, 0.25]
# [0.8, 0.1, 0.35]
# [0.7, 0.06, 0.1]
# 原V1 B40,W131
# [0.688, 0.021, 0.101]
# [0.780, 0.023, 0.222]
# [0.771, 0.026, 0.212]
# [0.733, 0.022, 0.354] False max 0.425 0.488 0.247   P
# V2 B48 W131
# [0.758, 0.184, 0.285] False 0.436,0.484,0.236
# [0.693, 0.262, 0.306] False 0.428,0.482,0.231
# [0.870, 0.163, 0.129] False 0.432 0.486 0.238
# [0.825, 0.190, 0.334] False 0.441 0.493 0.239
# [0.860, 0.252, 0.343] False 0.437 0.492 0.242
# [0.887, 0.294, 0.287] False 0.439 0.491 0.241
# [0.897, 0.245, 0.278] False 0.432 0.489 0.240
# [0.893, 0.151, 0.146] False 0.433 0.485 0.240

# centerness_guide B48 W131
# [0.825, 0.190, 0.334] e18 False 0.430 0.488 0.229
# [0.853, 0.194, 0.247] e18 False 0.431 0.478 0.224
# [0.852, 0.105, 0.318] e18 False 0.430 0.483 0.223
# [0.899, 0.166, 0.208] e18 False 0.432 0.481 0.223
# [0.832, 0.213, 0.241] e18 False 0.432 0.483 0.222
# [0.894, 0.121, 0.339] e18 False 0.431 0.487 0.226
# [0.889, 0.145, 0.315] e18 False 0.431 0.484 0.228
# [0.802, 0.275, 0.266] e18 False 0.429 0.480 0.223
# [0.771, 0.243, 0.314] e18 False 0.431 0.488 0.227
# [0.803, 0.242, 0.293]

# IOU_guide B48 W131
# [0.832, 0.213, 0.241] e19 False 0.433 0.487 0.250
# [0.832, 0.213, 0.241] e20 False 0.435 0.484 0.2
# [0.762, 0.225, 0.320] e20 False 0.439	0.489 0.247

# IOU_guide B48 W131 QFL DFL LogIOU
# [0.800, 0.076, 0.314] e17 False 0.430 0.490 0.248
# [0.800, 0.076, 0.314] e18 False 0.441 0.491 0.244

# __C.HP_SEARCH.GOT10k = [0.800, 0.076, 0.314]
# lr策
# [0.997, 0.164, 0.325] e20    0.461	0.515	0.272
# [0.997, 0.164, 0.325] e18    0.446	0.496	0.254
# [0.994, 0.174, 0.296] e18    0.460    0.511   0.270
# [0.994, 0.174, 0.296] e20    0.460    0.514   0.274
# [0.685, 0.012, 0.357] e20    0.446	0.494	0.254
# [0.685, 0.012, 0.357] e18    0.435	0.482	0.244
# [0.685, 0.012, 0.357] e18    0.447	0.449	0.253
# [0.973, 0.122, 0.242] e20    0.460	0.516	0.268
# [0.973, 0.122, 0.242] e18    0.458	0.508	0.266
# [0.739, 0.000, 0.439] e18    0.443	0.488	0.251
# [0.739, 0.000, 0.439] e20    0.442	0.496	0.253
# [0.806, 0.236, 0.404] e18    0.445	0.492	0.251
# [0.911, 0.030, 0.405] e20    0.445	0.498	0.255
# [0.911, 0.030, 0.405] e20    0.455	0.507	0.264
# [0.890, 0.014, 0.460] e20    0.453	0.504	0.260
# [0.863, 0.148, 0.366] e20    0.446	0.495	0.256
# [0.885, 0.000, 0.365] e20    0.448	0.504	0.266
# [0.818, 0.058, 0.440] e20    0.442	0.494	0.255
# [0.700, 0.006, 0.318] e20    0.450	0.501	0.251
# [0.759, 0.092, 0.255] e20    0.456	0.503	0.255
# 0.803, 0.134, 0.325

# epoch50
# [0.997, 0.164, 0.325] e15    0.454	0.512	0.285
# [0.997, 0.164, 0.325] e16    0.456	0.513	0.273
# [0.997, 0.164, 0.325] e17    0.482	0.548	0.304
# [0.997, 0.164, 0.325] e18    0.480	0.552	0.302
# [0.997, 0.164, 0.325] e22    0.479	0.548	0.312
# [0.997, 0.164, 0.325] e28    0.491	0.560	0.325

# 'window_influence': 0.08117466253542097, 'penalty_k': 0.1403396387944281, 'scale_lr': 0.9734885470598306
# 'window_influence': 0.5884544668858452, 'penalty_k': 0.8667061885105815, 'scale_lr': 0.9858079315178817
# 'window_influence': 0.427755200814682, 'penalty_k': 0.3846567899965575, 'scale_lr': 0.7077376553270803
# 'window_influence': 0.38979371719312383, 'penalty_k': 0.39639535558319133, 'scale_lr': 0.6635625431786243
# 'window_influence': 0.3812035200464907, 'penalty_k': 0.17706268309840606, 'scale_lr': 0.6571435661531597
# 'window_influence': 0.36359163570902686, 'penalty_k': 0.07694448352696925, 'scale_lr': 0.6312591579169273
# 'window_influence': 0.1925256837569455, 'penalty_k': 0.19943762152038605, 'scale_lr': 0.569934390911942
# 'window_influence': 0.13435145026040762, 'penalty_k': 0.12522478099826087, 'scale_lr': 0.6208921507787302}
# 'window_influence': 0.3704920731480298, 'penalty_k': 0.02375191930956197, 'scale_lr': 0.6548054664469103
# 'window_influence': 0.35146902587510054, 'penalty_k': 0.01351392338338195, 'scale_lr': 0.5161634149953508
# 'window_influence': 0.24795884597539347, 'penalty_k': 0.12750197830413373, 'scale_lr': 0.5175855507896745

# 'window_influence': 0.2977144090766529, 'penalty_k': 0.05574819829989971, 'scale_lr': 0.9030011258116851
# 'window_influence': 0.17751336028078094, 'penalty_k': 0.5402454211573301, 'scale_lr': 0.948357931302235
# 'window_influence': 0.02143885456973424, 'penalty_k': 0.04835740959713185, 'scale_lr': 0.9010898160494438
# 'window_influence': 0.14282745185135065, 'penalty_k': 0.06602552871908007, 'scale_lr': 0.9031723519499495
# 'window_influence': 0.041246793758884376, 'penalty_k': 0.21302502394144, 'scale_lr': 0.9694288518564936
# 'window_influence': 0.15575679791591623, 'penalty_k': 0.02551460581070528, 'scale_lr': 0.8945315093445996
# 'window_influence': 0.09291829346806847, 'penalty_k': 0.06557917500039832, 'scale_lr': 0.8928278848908989
# 'window_influence': 0.09771186710047457, 'penalty_k': 0.06974896591186995, 'scale_lr': 0.9274229909116375
# 'window_influence': 0.14828160241645433, 'penalty_k': 0.13852741560159718, 'scale_lr': 0.9132707585231064
# 'window_influence': 0.12730412805066801, 'penalty_k': 0.1306977209261406, 'scale_lr': 0.9909543571070936
# 'window_influence': 0.0352451455102014, 'penalty_k': 0.1325167606671278, 'scale_lr': 0.9966517641290329
# 'window_influence': 0.17135652641203966, 'penalty_k': 0.1073593953240638, 'scale_lr': 0.9506167653253417
# 'window_influence': 0.2012414646503163, 'penalty_k': 0.18680331815485257, 'scale_lr': 0.9500509203530781
# 'window_influence': 0.17023609180165658, 'penalty_k': 0.10628919969505705, 'scale_lr': 0.9999647725199097
# 'window_influence': 0.1102404711494066, 'penalty_k': 0.1635951380076639, 'scale_lr': 0.9695426497178045
# 'window_influence': 0.12169096047373583, 'penalty_k': 0.20396070210272288, 'scale_lr': 0.9322696457766909
# 'window_influence': 0.27534727861419717, 'penalty_k': 0.16296931442904242, 'scale_lr': 0.8678577933738922
# 'window_influence': 0.3245793040117825, 'penalty_k': 0.040888223417585745, 'scale_lr': 0.8125671300419843
# 'window_influence': 0.2761907853622399, 'penalty_k': 0.11687396916960366, 'scale_lr': 0.9000879158926705
# 'window_influence': 0.286200672458845, 'penalty_k': 0.14321657985993683, 'scale_lr': 0.96487862480215
# 'window_influence': 0.32615227699226784, 'penalty_k': 0.116405806068815, 'scale_lr': 0.9077608558093122
# 'window_influence': 0.2537125121246108, 'penalty_k': 0.12009565309346532, 'scale_lr': 0.9716253831433238
# 'window_influence': 0.2609922482703674, 'penalty_k': 0.14440015837066655, 'scale_lr': 0.9788650445889904
# 'window_influence': 0.2985430827776566, 'penalty_k': 0.11282714996828953, 'scale_lr': 0.9450747637411142
# 'window_influence': 0.2527547253394025, 'penalty_k': 0.176729998657605, 'scale_lr': 0.9664466462859548



# 0.997, 0.184, 0.273
# 0.9664466462859548, 0.176729998657605, 0.2527547253394025
__C.HP_SEARCH.GOT10k = [0.9664466462859548, 0.176729998657605, 0.2527547253394025]


# 'window_influence': 0.35466779590621095, 'penalty_k': 0.029082391985126483, 'scale_lr': 0.6012992916893327          0.629
#  window_influence: 0.37754251877036743, penalty_k: 0.01161153047601865, scale_lr: 0.24957306416597028,         AUC: 0.632
# checkpoint_e14 window_influence: 0.19817652920292675, penalty_k: 0.03601090114547258, scale_lr: 0.23906507078071881, AUC: 0.634
__C.HP_SEARCH.UAV123 = [0.2390650707807188, 0.036010901145472576, 0.19817652920292675]

__C.HP_SEARCH.LaSOT = [0.87748401336284709, 0.14794071292069522, 0.17530970310568331]

# VOT2018 |    Ours    |  0.598   |   0.239    |    51.0     | 0.410 |
__C.HP_SEARCH.VOT2018 = [0.5003972357626941, 0.15761085296157645, 0.34783423829042132]
# 0.5305405311289436, 0.1607862893276971, 0.35471211022039545
__C.HP_SEARCH.VOT2019 = [0.5003972357626941, 0.15761085296157645, 0.34783423829042132]
