"""
idea is to take couple best models use their outputs as additional channels 
connect them all through shallow unet and write their output
"""
### Define Data Handling

import concurrent.futures
import functools
import glob
import importlib.util
import itertools
import json
import math
import multiprocessing
import multiprocessing as mp
import operator
import os
import os.path
import shutil
import sys
import tempfile
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from glob import glob
from os import path as pathOs
from os.path import basename, dirname, exists, isdir, join, split
from pathlib import Path
#from picai_eval.picai_eval import evaluate_case
from statistics import mean
from typing import (Callable, Dict, Hashable, Iterable, List, Optional,
                    Sequence, Sized, Tuple, Union)

import gdown
import matplotlib.pyplot as plt
import model.DataModule as DataModule
import model.LigtningModel as LigtningModel
# import preprocessing.transformsForMain
# import preprocessing.ManageMetadata
import model.unets as unets
import monai
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio
import torchio as tio
import torchmetrics
# import preprocessing.semisuperPreprosess
from model import transformsForMain as transformsForMain
from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import (CacheDataset, Dataset, PersistentDataset,
                        decollate_batch, list_data_collate)
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
# lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
from monai.metrics import (ConfusionMatrixMetric, DiceMetric,
                           HausdorffDistanceMetric, SurfaceDistanceMetric,
                           compute_confusion_matrix_metric,
                           do_metric_reduction, get_confusion_matrix)
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers import Norm
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.networks.nets import UNet
from monai.transforms import (AddChanneld, AsDiscrete, AsDiscreted, Compose,
                              ConcatItemsd, CropForegroundd, DivisiblePadd,
                              EnsureChannelFirstd, EnsureType, EnsureTyped,
                              Invertd, LoadImaged, MapTransform, Orientationd,
                              Rand3DElasticd, RandAdjustContrastd, RandAffined,
                              RandCoarseDropoutd, RandCropByPosNegLabeld,
                              RandFlipd, RandGaussianNoised,
                              RandGaussianSmoothd, RandRicianNoised,
                              RandSpatialCropd, RepeatChanneld, Resize,
                              Resized, ResizeWithPadOrCropd, SaveImaged,
                              ScaleIntensityRanged, SelectItemsd, Spacingd,
                              SpatialPadd)
from monai.utils import alias, deprecated_arg, export, set_determinism
from optuna.integration import PyTorchLightningPruningCallback
from picai_eval import evaluate
from picai_eval.analysis_utils import (calculate_dsc, calculate_iou,
                                       label_structure, parse_detection_map)
from picai_eval.eval import evaluate_case
from picai_eval.image_utils import (read_label, read_prediction,
                                    resize_image_with_crop_or_pad)
from picai_eval.metrics import Metrics
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from report_guided_annotation import extract_lesion_candidates
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from torch.nn.intrinsic.qat import ConvBnReLU3d
from torch.utils.cpp_extension import load
from torch.utils.data import DataLoader, random_split
from torchmetrics import Precision
from torchmetrics.functional import precision_recall
from tqdm import tqdm
def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res

semisuperPreprosess =loadLib("semisuperPreprosess", "/home/sliceruser/locTemp/piCaiCode/preprocessing/semisuperPreprosess.py")


def getUnetA(dropout,input_image_size,in_channels,out_channels):
    return unets.UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)],
        channels=[32, 64, 128, 256, 512],
        num_res_units= 0,
        act = (Act.PRELU, {"init": 0.2}),
        norm= (Norm.BATCH, {}),
        dropout= dropout
    )
def getUnetB(dropout,input_image_size,in_channels,out_channels ):
    return unets.UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
        channels=[32, 64, 128, 256, 512, 1024],
        num_res_units= 0,
        act = (Act.PRELU, {"init": 0.2}),
        norm= (Norm.BATCH, {}),
        dropout= dropout
    )

def getUnetC(dropout,input_image_size,in_channels,out_channels ):
    return unets.UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2),(1, 1, 1)],
        channels=[32, 64, 128, 256, 512, 1024,512],
        num_res_units= 0,
        act = (Act.PRELU, {"init": 0.2}),
        norm= (Norm.BATCH, {}),
        dropout= dropout
    )


def getAhnet(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.AHNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        psp_block_num=3   )

def getSegResNet(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout,
    )
def getSegResNetVAE(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.SegResNetVAE(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout,
        input_image_size=input_image_size#torch.Tensor(input_image_size)

    )


def getAttentionUnet(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.AttentionUnet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        # img_size=input_image_size,
        strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)],
        channels=[32, 64, 128, 256, 512],
        dropout=dropout
    )
def getSwinUNETR(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.SwinUNETR(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=input_image_size
    )
def getVNet(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.VNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout_prob=dropout
    )
def getViTAutoEnc(dropout,input_image_size,in_channels,out_channels):
    return monai.networks.nets.ViTAutoEnc(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=input_image_size,
        patch_size=(16,16,16)
    )
#batch sizes


def getOptNAdam(lr):
    return torch.optim.NAdam(lr=lr)
def getOptions():
#getViTAutoEnc,getAhnet,getSegResNetVAE,getAttentionUnet,getSwinUNETR,getSegResNet,getVNet,getUnetB
    return {

    # "models":[getUnetA,getUnetB,getVNet,getSegResNet],
    #"models":[getUnetA,getUnetB,getUnetC],# ,getSegResNet,getSwinUNETR,getUnetA,getUnetB,getVNet,getUnetC
    "models":[getVNet,getSegResNet,getSwinUNETR,getUnetA,getUnetB,getUnetC],# ,getSegResNet,getSwinUNETR,getUnetA,getUnetB,getUnetC
    #getUnetA,getUnetB,getSegResNet,   getVNet
    "regression_channels":[[32,64,128],[10,16,32]], #,[10,16,32],
    #"regression_channels":[[32,64,128]], #,
    "optimizer_class": [getOptNAdam] ,# ,torch.optim.LBFGS optim.AggMo,   look in https://pytorch-optimizer.readthedocs.io/en/latest/api.html
    # "centerCropSize":[(256, 256,32)],
    "spacing_keyword" : ["_one_spac_c"]#, "_one_and_half_spac_c", "_two_spac_c"
    #"spacing_keyword" : ["_half_spac_c"]
    }
