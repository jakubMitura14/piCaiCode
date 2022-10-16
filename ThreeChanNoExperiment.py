### Define Data Handling

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
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset,Dataset,PersistentDataset, list_data_collate, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import time

from datetime import datetime
import os
import tempfile
from glob import glob
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from picai_eval import evaluate
#from picai_eval.picai_eval import evaluate_case
from statistics import mean
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter
import tempfile
import shutil
from os import path as pathOs
from os.path import basename, dirname, exists, isdir, join, split
import torch.nn as nn
import torch.nn.functional as F

from monai.metrics import (DiceMetric, HausdorffDistanceMetric,
                           SurfaceDistanceMetric)
from torchmetrics import Precision
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
)
import torchio
import importlib.util
import sys
import warnings
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export
import functools
import operator
from torch.nn.intrinsic.qat import ConvBnReLU3d

import multiprocessing as mp
import time



import time
from functools import partial
from torchmetrics.functional import precision_recall
from torch.utils.cpp_extension import load
import torchmetrics
# lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
from monai.metrics import (
    ConfusionMatrixMetric,
    compute_confusion_matrix_metric,
    do_metric_reduction,
    get_confusion_matrix,
)

import concurrent.futures
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (Callable, Dict, Hashable, Iterable, List, Optional, Sized,
                    Tuple, Union)

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

from picai_eval.analysis_utils import (calculate_dsc, calculate_iou,
                                       label_structure, parse_detection_map)
from picai_eval.image_utils import (read_label, read_prediction,
                                    resize_image_with_crop_or_pad)
from picai_eval.metrics import Metrics

from picai_eval.eval import evaluate_case
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
import functools
import multiprocessing as mp
import os
import os.path
monai.utils.set_determinism()
from functools import partial
from pytorch_lightning.loggers import CometLogger
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint

# import preprocessing.transformsForMain
# import preprocessing.ManageMetadata
import model.unets as unets
import model.DataModule as DataModule
import model.LigtningModel as LigtningModel
# import preprocessing.semisuperPreprosess

def getParam(trial,options,key):
    """
    given integer returned from experiment 
    it will look into options dictionary and return required object
    """
    lenn= len(options[key])
    #print(f"  ")
    integerr=trial.suggest_int(key, 0, lenn-1)

    return options[key][integerr]






def getModel(trial,df,experiment_name,dummyDict,options,percentSplit, in_channels
    ,out_channels,picaiLossArr_auroc_final,picaiLossArr_AP_final,picaiLossArr_score_final, dice_final
    ,persistent_cache,expId,checkPointPath ):        

    #basically id of trial 
    
    spacing_keyword=getParam(trial,options,"spacing_keyword")
    label_name=f"label_{spacing_keyword}fi"
    
    
    t2wColName="t2w"+spacing_keyword+"cropped"
    adcColName="adc"+spacing_keyword+"cropped"
    hbvColName="hbv"+spacing_keyword+"cropped"
    dummyLabelPath,img_size=dummyDict[spacing_keyword]
    chan3_col_name="joined"+spacing_keyword+"cropped"
    label_name_val=label_name
    chan3_col_name_val=chan3_col_name

 
    # df=df.loc[df[label_name_val] != ' ']
    df=df.loc[df[t2wColName] != ' ']
    df=df.loc[df[adcColName] != ' ']
    df=df.loc[df[hbvColName] != ' ']

    comet_logger = CometLogger(
        api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
        #workspace="OPI", # Optional
        project_name=experiment_name, # Optional
        #experiment_name="baseline" # Optional
    )
    
    data = DataModule.PiCaiDataModule(
        trainSizePercent=percentSplit,# 
        df= df,
        batch_size=12,#
        num_workers=os.cpu_count(),#os.cpu_count(),
        drop_last=False,#True,
        #we need to use diffrent cache folders depending on weather we are dividing data or not
        chan3_col_name =chan3_col_name,
        chan3_col_name_val=chan3_col_name_val,
        label_name_val=label_name_val,
        label_name=label_name
        ,t2wColName=t2wColName
        ,adcColName=adcColName
        ,hbvColName=hbvColName
        #maxSize=maxSize
        ,RandAdjustContrastd_prob=trial.suggest_float("RandAdjustContrastd_prob", 0.0, 0.6)
        ,RandGaussianSmoothd_prob=trial.suggest_float("RandGaussianSmoothd_prob", 0.0, 0.6)
        ,RandRicianNoised_prob=trial.suggest_float("RandRicianNoised_prob", 0.0, 0.6)
        ,RandFlipd_prob=trial.suggest_float("RandFlipd_prob", 0.0, 0.6)
        ,RandAffined_prob=trial.suggest_float("RandAffined_prob", 0.0, 0.6)
        ,RandomElasticDeformation_prob=trial.suggest_float("RandomElasticDeformation_prob", 0.0, 0.6)
        ,RandomAnisotropy_prob=trial.suggest_float("RandomAnisotropy_prob", 0.0, 0.6)
        ,RandomMotion_prob=trial.suggest_float("RandomMotion_prob", 0.0, 0.6)
        ,RandomGhosting_prob=trial.suggest_float("RandomGhosting_prob", 0.0, 0.6)
        ,RandomSpike_prob=trial.suggest_float("RandomSpike_prob", 0.0, 0.6)
        ,RandomBiasField_prob=trial.suggest_float("RandomBiasField_prob", 0.0, 0.6)
        ,persistent_cache=persistent_cache
    )

    dropout= trial.suggest_float("dropout", 0.0,0.6)
    # data.prepare_data()
    # data.setup()
    net= getParam(trial,options,"models") #options["models"][0]#   
    net=net(dropout,img_size,in_channels,out_channels)

    optimizer_class= torch.optim.NAdam
    regression_channels=getParam(trial,options,"regression_channels")
    to_onehot_y_loss= False
    

    model = LigtningModel.Model(
         net=net,
        criterion=  monai.losses.FocalLoss(include_background=False, to_onehot_y=to_onehot_y_loss),# Our seg labels are single channel images indicating class index, rather than one-hot
        learning_rate=trial.suggest_float("learning_rate", 1e-6, 1e-4),
        optimizer_class= optimizer_class,
        picaiLossArr_auroc_final=picaiLossArr_auroc_final,
        picaiLossArr_AP_final=picaiLossArr_AP_final,
        picaiLossArr_score_final=picaiLossArr_score_final,
        regression_channels=regression_channels,
        trial=trial
        ,dice_final=dice_final
    )

    checkpoint_callback = ModelCheckpoint(dirpath= checkPointPath,mode='max', save_top_k=1, monitor="dice")
    stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs=trial.suggest_float("swa_lrs", 1e-6, 1e-4))
    optuna_prune=PyTorchLightningPruningCallback(trial, monitor="dice")     
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='dice',
        patience=15,
        mode="min",
        #divergence_threshold=(-0.1)
    )

    trainer = pl.Trainer(
        #accelerator="cpu", #TODO(remove)
        max_epochs=4000,
        #gpus=1,
        #precision=experiment.get_parameter("precision"), 
        callbacks=[ checkpoint_callback,stochasticAveraging,early_stopping ], #optuna_prune
        logger=comet_logger,
        accelerator='auto',
        devices='auto',       
        default_root_dir= "/home/sliceruser/locTemp/lightning_logs",
        # auto_scale_batch_size="binsearch",
        # auto_lr_find=True,
        check_val_every_n_epoch=50,
        accumulate_grad_batches= 2,
        gradient_clip_val=  0.9 ,#experiment.get_parameter("gradient_clip_val"),# 0.5,2.0
        log_every_n_steps=3
        #strategy='dp'
    )
    #trainer.logger._default_hp_metric = False

    return (trainer, model, data)


 