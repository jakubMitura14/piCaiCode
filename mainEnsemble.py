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
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
import importlib.util
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import sys
def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res

semisuperPreprosess =loadLib("semisuperPreprosess", "/home/sliceruser/locTemp/piCaiCode/preprocessing/semisuperPreprosess.py")
import hyperParamHelper

monai.utils.set_determinism()
import model.LigtningModel as LigtningModel

def getTrialNumberFromPath(checkpointPath):
    # fullDir=os.path.dirname(checkpointPath)
    
    res= int(Path(checkpointPath).stem)
    print(f" getTrialNumberFromPath {res} {checkpointPath} ")
    return res
def loadModel(checkPointPath,trials,options,train_transforms,train_transforms_noLabel,val_transforms):
    trialNum=getTrialNumberFromPath(checkPointPath)
    trial=trials[trialNum]
    trialProp=trial.params    
    picaiLossArr_auroc_final=[]
    picaiLossArr_AP_final=[]
    picaiLossArr_score_final=[]
    dice_final=[]
    netIndex=trialProp["models"]
    isVnet=(netIndex==0)
    in_channels=3
    out_channels=2
    if(isVnet):
        in_channels=4
    spacing_keyword=options["spacing_keyword"][0]
    dummyLabelPath,img_size=dummyDict[spacing_keyword]
    label_name=f"label_{spacing_keyword}fi"
    
    t2wColName="t2w"+spacing_keyword+"cropped"
    adcColName="adc"+spacing_keyword+"cropped"
    hbvColName="hbv"+spacing_keyword+"cropped"
    chan3_col_name="joined"+spacing_keyword+"cropped"
    label_name_val=label_name
    chan3_col_name_val=chan3_col_name

    net = options["models"][netIndex]
    net=net(0.0,img_size,in_channels,out_channels)
    regr_chan_index=trialProp["regression_channels"]

    return LigtningModel.Model.load_from_checkpoint(checkPointPath
        , net=net
        , criterion=monai.losses.FocalLoss(include_background=False, to_onehot_y=False)
        , learning_rate=1e-5
        , optimizer_class=torch.optim.NAdam
        ,picaiLossArr_auroc_final=picaiLossArr_auroc_final
        ,picaiLossArr_AP_final=picaiLossArr_AP_final
        ,picaiLossArr_score_final=picaiLossArr_score_final
        ,regression_channels= options["regression_channels"][regr_chan_index]
        ,trial=trial
        ,dice_final=dice_final
        ,trainSizePercent=0.85
        ,batch_size=2
        ,num_workers=os.cpu_count()
        ,drop_last=False
        ,df=df
        ,chan3_col_name=chan3_col_name
        ,chan3_col_name_val=chan3_col_name_val
        ,label_name=label_name
        ,label_name_val=label_name_val
        ,t2wColName=t2wColName
        ,adcColName=adcColName
        ,hbvColName=hbvColName
        ,RandAdjustContrastd_prob=0.0
        ,RandGaussianSmoothd_prob=0.0
        ,RandRicianNoised_prob=0.0
        ,RandFlipd_prob=0.0
        ,RandAffined_prob=0.0
        ,RandomElasticDeformation_prob=0.0
        ,RandomAnisotropy_prob=0.0
        ,RandomMotion_prob=0.0
        ,RandomGhosting_prob=0.0
        ,RandomSpike_prob=0.0
        ,RandomBiasField_prob=0.0
        ,persistent_cache=persistent_cache
        ,spacing_keyword=spacing_keyword
        ,netIndex=netIndex
        ,regr_chan_index=regr_chan_index
        ,isVnet=isVnet
        ,train_transforms=train_transforms
        ,train_transforms_noLabel=train_transforms_noLabel
        ,val_transforms=val_transforms
        ).modelRegression.cuda()

def forwardLoadedModel(model,x):
    segmMap,regr = model(x)
    return segmMap[:,1,:,:,:]

class UNetToEnsemble(nn.Module):
    def __init__(self,
        modelPaths,
        study,
        options,train_transforms,train_transforms_noLabel,val_transforms
    ) -> None:
        super().__init__()
        self.baseUnet= unets.UNet(
            spatial_dims=3,
            in_channels=3+len(modelPaths),
            out_channels=2,
            strides=[(2, 2, 2), (1, 2, 2)],
            channels=[32, 64, 128],
            num_res_units= 0,
            act = (Act.PRELU, {"init": 0.2}),
            norm= (Norm.BATCH, {}))
        study.trials
        self.loadedModels= list(map(partial(loadModel,trials=study.trials,options=options
        ,train_transforms=train_transforms
            ,train_transforms_noLabel=train_transforms_noLabel,val_transforms=val_transforms), modelPaths))
        
    

    def forward(self, x):
        from_models =torch.stack( list(map(lambda model: forwardLoadedModel(model,x),self.loadedModels)),1)
        # print(f"from_models {from_models.size()}  x {x.size()} ")
        stackedInput=torch.cat((x,from_models),1)

        #print(f"segmMap  {segmMap}")
        return self.baseUnet(stackedInput)

# model = LigtningModel.Model.load_from_checkpoint("/path/to/checkpoint.ckpt")


def getUnetEnsemble(modelPaths,study,options,train_transforms,train_transforms_noLabel,val_transforms):
    return UNetToEnsemble(modelPaths,study,options,train_transforms,train_transforms_noLabel,val_transforms)



def get_transforms_no_label_ensembl():

    val_transforms = Compose(
        [
            LoadImaged(keys=["t2w","hbv","adc" ],reader="ITKReader"),
            EnsureChannelFirstd(keys=["t2w","hbv","adc" ]),
            EnsureTyped(keys=["t2w","hbv","adc" ]),
            ConcatItemsd(["t2w","hbv","adc" ], "chan3_col_name"),
            SelectItemsd(keys=["labelB","t2wb","chan3_col_name","study_id","num_lesions_to_retain","isAnythingInAnnotated"]),
        ]
    )
    return val_transforms
def get_transforms_label_ensembl():

    val_transforms = Compose(
        [
            LoadImaged(keys=["t2w","hbv","adc" ,"label_name"],reader="ITKReader"),
            EnsureChannelFirstd(keys=["t2w","hbv","adc" ,"label_name"]),
            EnsureTyped(keys=["t2w","hbv","adc" ,"label_name"]),
            transformsForMain.standardizeLabels(keys=["label_name"]),
            AsDiscreted(keys=["label_name"], to_onehot=2),
            ConcatItemsd(["t2w","hbv","adc" ], "chan3_col_name"),
            SelectItemsd(keys=["labelB","t2wb","chan3_col_name","label_name","study_id","num_lesions_to_retain","isAnythingInAnnotated"]),
         ]
    )
    return val_transforms




def getEnsemble(df,experiment_name,dummyDict,options,percentSplit
,picaiLossArr_auroc_final,picaiLossArr_AP_final,picaiLossArr_score_final, dice_final
    ,persistent_cache,checkPointPath_to_save
    ,checkpointPaths_to_load,regression_channelsNum,study):        

    spacing_keyword=options["spacing_keyword"][0]
    dummyLabelPath,img_size=dummyDict[spacing_keyword]


    in_channels=3+len(checkpointPaths_to_load)
    out_channels=2
    batch_size=2
    train_transforms_noLabel=get_transforms_no_label_ensembl()
    train_transforms=get_transforms_label_ensembl()
    val_transforms=train_transforms    

    net=getUnetEnsemble(checkpointPaths_to_load,study,options,train_transforms,train_transforms_noLabel,val_transforms)

    label_name=f"label_{spacing_keyword}fi"
   
   
    t2wColName="t2w"+spacing_keyword+"cropped"
    adcColName="adc"+spacing_keyword+"cropped"
    hbvColName="hbv"+spacing_keyword+"cropped"
    chan3_col_name="joined"+spacing_keyword+"cropped"
    label_name_val=label_name
    chan3_col_name_val=chan3_col_name

 
    df=df.loc[df[t2wColName] != ' ']
    df=df.loc[df[adcColName] != ' ']
    df=df.loc[df[hbvColName] != ' ']

    comet_logger = CometLogger(
        api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
        #workspace="OPI", # Optional
        project_name=experiment_name, # Optional
    )
    
    optimizer_class= torch.optim.NAdam
    regr_chan_index=regression_channelsNum
    regression_channels=options["regression_channels"][regression_channelsNum]
    to_onehot_y_loss= False    
    RandAdjustContrastd_prob=0.0
    RandGaussianSmoothd_prob=0.0 #trial.suggest_float("RandGaussianSmoothd_prob", 0.0, 0.6)
    RandRicianNoised_prob=0.0#trial.suggest_float("RandRicianNoised_prob", 0.0, 0.9)
    RandFlipd_prob=0.0#trial.suggest_float("RandFlipd_prob", 0.0, 0.9)
    RandAffined_prob=0.0#trial.suggest_float("RandAffined_prob", 0.0, 0.9)
    RandomElasticDeformation_prob=0.0#trial.suggest_float("RandomElasticDeformation_prob", 0.0, 0.9)
    RandomAnisotropy_prob=0.0#trial.suggest_float("RandomAnisotropy_prob", 0.0, 0.9)
    RandomMotion_prob=0.0#trial.suggest_float("RandomMotion_prob", 0.0, 0.6)
    RandomGhosting_prob=0.0#trial.suggest_float("RandomGhosting_prob", 0.0, 0.6)
    RandomSpike_prob=0.0#trial.suggest_float("RandomSpike_prob", 0.0, 0.6)
    RandomBiasField_prob=0.0#trial.suggest_float("RandomBiasField_prob", 0.0, 0.6)
    isVnet=False




    model = LigtningModel.Model(
        net=net
        ,criterion=  monai.losses.FocalLoss(include_background=False, to_onehot_y=to_onehot_y_loss)# Our seg labels are single channel images indicating class index, rather than one-hot
        ,learning_rate=1e-5#trial.suggest_float("learning_rate", 1e-6, 1e-4),
        ,optimizer_class= optimizer_class
        ,picaiLossArr_auroc_final=picaiLossArr_auroc_final
        ,picaiLossArr_AP_final=picaiLossArr_AP_final
        ,picaiLossArr_score_final=picaiLossArr_score_final
        ,regression_channels=regression_channels
        ,trial=[]
        ,dice_final=dice_final
        ,trainSizePercent=percentSplit
        ,df= df
        ,batch_size=batch_size
        ,num_workers=(os.cpu_count())
        ,drop_last=False
        ,chan3_col_name =chan3_col_name
        ,chan3_col_name_val=chan3_col_name_val
        ,label_name_val=label_name_val
        ,label_name=label_name
        ,t2wColName=t2wColName
        ,adcColName=adcColName
        ,hbvColName=hbvColName
        ,RandAdjustContrastd_prob=RandAdjustContrastd_prob#trial.suggest_float("RandAdjustContrastd_prob", 0.0, 0.9)
        ,RandGaussianSmoothd_prob=RandGaussianSmoothd_prob#0.0 #trial.suggest_float("RandGaussianSmoothd_prob", 0.0, 0.6)
        ,RandRicianNoised_prob=RandRicianNoised_prob#trial.suggest_float("RandRicianNoised_prob", 0.0, 0.9)
        ,RandFlipd_prob=RandFlipd_prob#trial.suggest_float("RandFlipd_prob", 0.0, 0.9)
        ,RandAffined_prob=RandAffined_prob#trial.suggest_float("RandAffined_prob", 0.0, 0.9)
        ,RandomElasticDeformation_prob=RandomElasticDeformation_prob#trial.suggest_float("RandomElasticDeformation_prob", 0.0, 0.9)
        ,RandomAnisotropy_prob=RandomAnisotropy_prob#0.0#trial.suggest_float("RandomAnisotropy_prob", 0.0, 0.9)
        ,RandomMotion_prob=RandomMotion_prob#0.0#trial.suggest_float("RandomMotion_prob", 0.0, 0.6)
        ,RandomGhosting_prob=RandomGhosting_prob#0.0#trial.suggest_float("RandomGhosting_prob", 0.0, 0.6)
        ,RandomSpike_prob=RandomSpike_prob#0.0#trial.suggest_float("RandomSpike_prob", 0.0, 0.6)
        ,RandomBiasField_prob=RandomBiasField_prob#0.0#trial.suggest_float("RandomBiasField_prob", 0.0, 0.6)
        ,persistent_cache=persistent_cache
        ,spacing_keyword=spacing_keyword
        ,netIndex=0
        ,regr_chan_index=regr_chan_index
        ,isVnet=isVnet
        ,train_transforms=train_transforms
        ,train_transforms_noLabel=train_transforms_noLabel
        ,val_transforms=val_transforms
        ,threshold='dynamic'
        ,toWaitForPostProcess=5
        ,toLogHyperParam=False
    )

    toMonitor="score_my"
    checkpoint_callback = ModelCheckpoint(dirpath= checkPointPath_to_save,mode='max', save_top_k=1, monitor=toMonitor)
    stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging(swa_lrs=1e-4)

    # early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    #     monitor=toMonitor,
    #     patience=10,
    #     mode="max",
    #     #divergence_threshold=(-0.1)
    # )

    trainer = pl.Trainer(
        #accelerator="cpu", #TODO(remove)
        max_epochs=1000,
        #gpus=1,
        #precision=16,#experiment.get_parameter("precision"), 
        callbacks=[ checkpoint_callback,stochasticAveraging ], #optuna_prune,early_stopping
        logger=comet_logger,
        accelerator='auto',
        devices='auto',       
        default_root_dir= "/home/sliceruser/locTemp/lightning_logs",
        # auto_scale_batch_size="binsearch",
        auto_lr_find=True,
        check_val_every_n_epoch=20,
        accumulate_grad_batches= 1,
        gradient_clip_val=  0.9 ,#experiment.get_parameter("gradient_clip_val"),# 0.5,2.0
        log_every_n_steps=5
        ,reload_dataloaders_every_n_epochs=1
        #strategy='dp'
    )
    #trainer.logger._default_hp_metric = False
    trainer.fit(model)
    return (trainer, model)

# git lfs push --all origin main:main https://github.com/jakubMitura14/piCaiCode.git


df = pd.read_csv("/home/sliceruser/data/metadata/processedMetaData_current_b.csv")

def getDummy(spac):
    label_name=f"label_{spac}" 
    print(df[label_name])
    imageRef_path=list(filter(lambda it: it!= " ", df[label_name].to_numpy()))[0]
    dummyLabelPath=f"/home/sliceruser/data/dummyData/zeroLabel{spac}.nii.gz"
    sizz=semisuperPreprosess.writeDummyLabels(dummyLabelPath,imageRef_path)
    img_size = (sizz[2],sizz[1],sizz[0])
    return(dummyLabelPath,img_size)

options = hyperParamHelper.getOptions()
spacings =  options['spacing_keyword']#["_half_spac_c", "_one_spac_c", "_one_and_half_spac_c", "_two_spac_c" ]# ,"_med_spac_b" #config['parameters']['spacing_keyword']["values"]
aa=list(map(getDummy  ,spacings  ))
# dummyDict={"_half_spac_c":aa[0], "_one_spac_c":aa[1], "_one_and_half_spac_c":aa[2], "_two_spac_c":aa[3]}
dummyDict={"_one_spac_c":aa[0]}

#from https://www.askpython.com/python/examples/python-directory-listing
def list_full_paths(directory):
    return [os.path.join(directory, file) for file in os.listdir(directory)]
 


picaiLossArr_auroc_final=[]
picaiLossArr_AP_final=[]
picaiLossArr_score_final=[]
dice_final=[]
persistent_cache=tempfile.mkdtemp()
experiment_name="ensemble1"
dummyDict=dummyDict
options=options
percentSplit=0.85
checkPointPath_to_save=f"/home/sliceruser/locTemp/checkPoints/{experiment_name}"
regression_channelsNum=1
checkpointPaths_to_load= list_full_paths('/home/sliceruser/locTemp/checkPointsIn/checkpoints')
print(f"checkpointPaths_to_load {checkpointPaths_to_load}")
studyNameToLoad="pic53"
study = optuna.create_study(
        study_name=studyNameToLoad
        ,sampler=optuna.samplers.NSGAIISampler()    
        ,pruner=optuna.pruners.HyperbandPruner()
        ,storage=f"mysql://root:jm@34.91.215.109:3306/{studyNameToLoad}"
        ,load_if_exists=True
        ,direction="maximize"
        )

getEnsemble(df,experiment_name,dummyDict,options,percentSplit
,picaiLossArr_auroc_final,picaiLossArr_AP_final,picaiLossArr_score_final, dice_final
    ,persistent_cache,checkPointPath_to_save, checkpointPaths_to_load
    ,regression_channelsNum,study)