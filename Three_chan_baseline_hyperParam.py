from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import time
from pathlib import Path
from datetime import datetime
import SimpleITK as sitk
from monai.utils import set_determinism
import math
import torch
from torch.utils.data import random_split, DataLoader
import monai
import gdown
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.data import CacheDataset,Dataset,PersistentDataset, list_data_collate, decollate_batch
from datetime import datetime
import os
import tempfile
from glob import glob
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from monai.networks.layers.factories import Act, Norm
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
from comet_ml import Optimizer

monai.utils.set_determinism()

import importlib.util
import sys

def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res

transformsForMain =loadLib("transformsForMain", "/home/sliceruser/data/piCaiCode/preprocessing/transformsForMain.py")
manageMetaData =loadLib("ManageMetadata", "/home/sliceruser/data/piCaiCode/preprocessing/ManageMetadata.py")
dataUtils =loadLib("dataUtils", "/home/sliceruser/data/piCaiCode/dataManag/utils/dataUtils.py")

unets =loadLib("unets", "/home/sliceruser/data/piCaiCode/model/unets.py")
DataModule =loadLib("DataModule", "/home/sliceruser/data/piCaiCode/model/DataModule.py")
LigtningModel =loadLib("LigtningModel", "/home/sliceruser/data/piCaiCode/model/LigtningModel.py")
import Three_chan_baseline
#####hyperparameters
# based on https://www.comet.com/docs/python-sdk/introduction-optimizer/
# We only need to specify the algorithm and hyperparameters to use:
config = {
    # We pick the Bayes algorithm:
    "algorithm": "bayes",

    # Declare  hyperparameters in 
    "parameters": {
        "loss": {"type": "discrete", "values": [monai.losses.FocalLoss(include_background=False, to_onehot_y=True)]},
        "stridesAndChannels": {"type": "discrete", "values": [{
                                                            "strides":[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
                                                            ,"channels":[32, 64, 128, 256, 512, 1024]
                                                            }]},
        "optimizer_class": {"type": "discrete", "values": [torch.optim.AdamW]},
        "num_res_units": {"type": "discrete", "values": [0]},
        "act": {"type": "discrete", "values": [(Act.PRELU, {"init": 0.2})]},
        "norm": {"type": "discrete", "values": [(Norm.INSTANCE, {})]},
        "dropout": {"type": "discrete", "values": [0.0]},
        "precision": {"type": "discrete", "values": [16]},
        "accumulate_grad_batches": {"type": "discrete", "values": [1]},
        "gradient_clip_val": {"type": "discrete", "values": [0.0]},
        "RandGaussianNoised_prob": {"type": "discrete", "values": [0.1]},
        "RandAdjustContrastd_prob": {"type": "discrete", "values": [0.1]},
        "RandGaussianSmoothd_prob": {"type": "discrete", "values": [0.1]},
        "RandRicianNoised_prob": {"type": "discrete", "values": [0.1]},
        "RandFlipd_prob": {"type": "discrete", "values": [0.1]},
        "RandAffined_prob": {"type": "discrete", "values": [0.1]},
        "RandCoarseDropoutd_prob": {"type": "discrete", "values": [0.1]},
        "is_whole_to_train": {"type": "discrete", "values": [True,False]},
        "dirs": {"type": "discrete", "values": [
                                                {
                                                "cache_dir":"/home/sliceruser/preprocess/monai_persistent_Dataset"
                                                ,"t2w_name":"t2w_med_spac"
                                                ,"adc_name":"registered_adc_med_spac"
                                                ,"hbv_name":"registered_hbv_med_spac"
                                                ,"label_name":"label_med_spac" 
                                                ,"metDataDir":"/home/sliceruser/data/metadata/processedMetaData_current.csv"
                                                }
                                                    ]},
    },

    # Declare what we will be optimizing, and how:
    "spec": {
    "metric": "val_mean_Dice_metr",
        "objective": "minimize",
    },
}

opt = Optimizer(config)

for experiment in opt.get_experiments(
        api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
        project_name="picai-hyperparam-search-01"):
    Three_chan_baseline.mainTrain(experiment)



