
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from comet_ml import Optimizer
import multiprocessing as mp

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
import torch_optimizer as optim
monai.utils.set_determinism()
import geomloss
from geomloss import SamplesLoss  # See also ImagesLoss, VolumesLoss
import optuna
from optuna.integration import PyTorchLightningPruningCallback
# from ray import air, tune
# from ray.air import session
# from ray.tune import CLIReporter
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
import importlib.util
import sys

def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res
import model.transformsForMain as transformsForMain
import model.DataModule as DataModule
manageMetaData =loadLib("ManageMetadata", "/home/sliceruser/data/piCaiCode/preprocessing/ManageMetadata.py")
dataUtils =loadLib("dataUtils", "/home/sliceruser/data/piCaiCode/dataManag/utils/dataUtils.py")

unets =loadLib("unets", "/home/sliceruser/data/piCaiCode/model/unets.py")
DataModule =loadLib("DataModule", "/home/sliceruser/data/piCaiCode/model/DataModule.py")
LigtningModel =loadLib("LigtningModel", "/home/sliceruser/data/piCaiCode/model/LigtningModel.py")
Three_chan_baseline =loadLib("Three_chan_baseline", "/home/sliceruser/data/piCaiCode/Three_chan_baseline.py")
semisuperPreprosess =loadLib("semisuperPreprosess", "/home/sliceruser/data/piCaiCode/preprocessing/semisuperPreprosess.py")





def getUnetA(dropout,input_image_size,in_channels,out_channels ):
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
def getUnetB(dropout,input_image_size,in_channels,out_channels):
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
        input_image_size=torch.Tensor(input_image_size)

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
#unet one spac sth like 16


def getOptNAdam(lr):
    return torch.optim.NAdam(lr=lr)

#getViTAutoEnc,getAhnet,getSegResNetVAE,getAttentionUnet,getSwinUNETR,getSegResNet,getVNet,getUnetB
options={

# "models":[getUnetA,getUnetB,getVNet,getSegResNet],
"models":[getUnetA],# getVNet,getSegResNet,getSwinUNETR
"regression_channels":[[10,16,32]], #[1,1,1],[2,4,8],

"optimizer_class": [getOptNAdam] ,# ,torch.optim.LBFGS optim.AggMo,   look in https://pytorch-optimizer.readthedocs.io/en/latest/api.html
"act":[(Act.PRELU, {"init": 0.2})],#,(Act.LEAKYRELU, {})                                         
"norm":[(Norm.BATCH, {}) ],
"centerCropSize":[(256, 256,32)],
#TODO() learning rate schedulers https://medium.com/mlearning-ai/make-powerful-deep-learning-models-quickly-using-pytorch-lightning-29f040158ef3
}





df = pd.read_csv("/home/sliceruser/data/metadata/processedMetaData_current_b.csv")

t2wColName='t2w'
adcColName='registered_'+'adc'
hbvColName='registered_'+'hbv'
# t2wColName="t2w"+spacing_keyword 
# adcColName="adc"+spacing_keyword
# hbvColName="hbv"+spacing_keyword

# df=df.head(40)
label_name="reSampledPath"
# label_name="label"+spacing_keyword
label_name_val=label_name
df=df.loc[df[label_name_val] != ' ']
df=df.loc[df[t2wColName] != ' ']
df=df.loc[df[adcColName] != ' ']
df=df.loc[df[hbvColName] != ' ']
df=df.loc[df['num_lesions_to_retain']>-1]#correct gleason ...

spacings =  ["_half_spac_c", "_one_spac_c", "_one_and_half_spac_c", "_two_spac_c" ]# ,"_med_spac_b" #config['parameters']['spacing_keyword']["values"]

def getDummy(spac):
    label_name=f"label{spac}_maxSize_" 
    imageRef_path=list(filter(lambda it: it!= '', df[label_name].to_numpy()))[0]
    dummyLabelPath=f"/home/sliceruser/data/dummyData/zeroLabel{spac}.nii.gz"
    sizz=semisuperPreprosess.writeDummyLabels(dummyLabelPath,imageRef_path)
    img_size = sizz#(sizz[2],sizz[1],sizz[0])
    return(dummyLabelPath,img_size)

aa=list(map(getDummy  ,spacings  ))
dummyDict={"_half_spac_c":aa[0] , "_one_spac_c":aa[1], "_one_and_half_spac_c":aa[2], "_two_spac_c":aa[3]}

physical_size =(81.0, 160.0, 192.0)#taken from picai used to crop image so only center will remain

experiment_name="picai_hp_35"
expId='a1'
Three_chan_baseline.mainTrain(options,df,physical_size,expId)

def objective(trial: optuna.trial.Trial) -> float:

    return Three_chan_baseline.mainTrain(trial,df,experiment_name,dummyDict
    ,num_workers,cpu_num ,default_root_dir,checkpoint_dir,options,num_cpus_per_worker)



study = optuna.create_study(
        study_name=experiment_name
        ,sampler=optuna.samplers.NSGAIISampler()    
        ,pruner=optuna.pruners.HyperbandPruner()
        ,storage="mysql://root@34.147.7.30:3306/picai_hp_35"
        ,load_if_exists=True
        #,storage="mysql://root@127.0.0.1:3306/picai_hp_35"
        )
        #mysql://root@localhost/example
study.optimize(objective, n_trials=40)

# for experiment in opt.get_experiments(
#         project_name="picai-hyperparam-search-43"):
#     print("******* new experiment *****")    
#     Three_chan_baseline.mainTrain(experiment,options,df,physical_size)
