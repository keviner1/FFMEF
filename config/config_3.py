#!/usr/local/bin/python
import sys
import os
import yaml
from yacs.config import CfgNode as CN


_C = CN()
# -----------------------------------------------------------------------------
#                              Server settings
# -----------------------------------------------------------------------------
_C.SERVER = CN()
_C.SERVER.gpus = 1
_C.SERVER.TRAIN_DATA = 'images/dataset/VIF/VIFB'
_C.SERVER.VAL_DATA = 'images/dataset/VIF/VIFB'
_C.SERVER.TEST_DATA = 'images/dataset/VIF/VIFB'
_C.SERVER.OUTPUT = 'output/job3'
_C.SERVER.train_len = 888888
_C.SERVER.val_len = 888888
_C.SERVER.test_len = 888888
# -----------------------------------------------------------------------------
#                            data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.crop_resize = (320,320)
_C.DATA.task = "VIF"
# -----------------------------------------------------------------------------
#                            Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.choice = "FFMEF"
_C.MODEL.channels = 4
# -----------------------------------------------------------------------------
#                           Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.NUM_WORKERS = 0
_C.TRAIN.EPOCHS = 30
_C.TRAIN.PRINT_FREQ = 20
_C.TRAIN.RESUME = ""

_C.TRAIN.amp = False
_C.TRAIN.loss = "GIF" #MEF-SSIM  ,  PMGI

# ----------------------------------LR scheduler-------------------------------
_C.TRAIN.LR_MODE = "step"
_C.TRAIN.BASE_LR = 0.001 #0.001
_C.TRAIN.LR_DECAY = 0.5
_C.TRAIN.LR_STEP = 30
_C.TRAIN.MILESTONES = [50, 100, 150, 250, 350, 500, 650, 800, 1000, 1300, 1600, 1900]
# -----------------------------------Optimizer-----------------------------
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
_C.TRAIN.OPTIMIZER.EPS = 1e-8
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9


def get_config():
    config = _C.clone()
    return config


