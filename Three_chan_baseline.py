
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

percentSplit=0.5

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


#comet_logger
### pytorch model mainly
# loss=monai.losses.FocalLoss(include_background=False, to_onehot_y=True)
# #loss=monai.losses.metrics.DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
# strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
# channels=[32, 64, 128, 256, 512, 1024]
# optimizer_class=torch.optim.AdamW
# num_res_units= 0
# act = (Act.PRELU, {"init": 0.2}) #LeakyReLU(negative_slope=0.1, inplace=True)
# norm= (Norm.INSTANCE, {}) #(Norm.INSTANCE, {"normalized_shape": (10, 10, 10)})#Norm.INSTANCE, #GroupNorm(1, 1, eps=1e-05, affine=False), LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
# dropout= 0.0 #0.2
# ### lightning trainer mainly
# precision=16# "bf16" 64
# max_epochs=30
# accumulate_grad_batches=1 # 3,5 ..
# gradient_clip_val=0.0# 0.5,2.0
# ## augmebtations mainly
# RandGaussianNoised_prob=0.1
# RandAdjustContrastd_prob=0.1
# RandGaussianSmoothd_prob=0.1
# RandRicianNoised_prob=0.1
# RandFlipd_prob=0.1
# RandAffined_prob=0.1
# RandCoarseDropoutd_prob=0.1
# is_whole_to_train =True
# ##diffrent definitions depending on preprocessing
# cache_dir="/home/sliceruser/preprocess/monai_persistent_Dataset"
# t2w_name= "t2w_med_spac"
# adc_name="registered_adc_med_spac"
# hbv_name="registered_hbv_med_spac"
# label_name="label_med_spac"

def getParam(experiment,options,key,df):
    """
    given integer returned from experiment 
    it will look into options dictionary and return required object
    """
    integerr=experiment.get_parameter(key)
    # print("keyy {key} ")
    # print(options[key])
    return options[key][integerr]




def mainTrain(experiment,options,df):
    finalLoss=[100]
    print("mmmmmmmmmmmmmmmmmm")
    #TODO(remove)
    # comet_logger = CometLogger(
    #     api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
    #     #workspace="OPI", # Optional
    #     project_name="picai_base_3Channels", # Optional
    #     #experiment_name="baseline" # Optional
    # )
    #############loading meta data 
    #maxSize=manageMetaData.getMaxSize(getParam(experiment,options,"dirs")["chan3_col_name"],df)
    # print(f"************    maxSize {maxSize}   ***************")
    spacing_keyword=experiment.get_parameter("spacing_keyword")
    sizeWord= experiment.get_parameter("sizeWord")
    chan3_col_name=f"t2w{spacing_keyword}_3Chan{sizeWord}" 
    label_name=f"label{spacing_keyword}{sizeWord}" 
    cacheDir =  f"/home/sliceruser/preprocess/monai_persistent_Dataset/{spacing_keyword}/{sizeWord}"
    maxSize=sizeWord=="_maxSize_"
    #print(f" fffffffffffff sizeWord {maxSize}")
    data = DataModule.PiCaiDataModule(
        df= df,
        batch_size=2,#
        trainSizePercent=percentSplit,# TODO(change to 0.7 or 0.8
        num_workers=os.cpu_count(),
        drop_last=False,#True,
        #we need to use diffrent cache folders depending on weather we are dividing data or not
        cache_dir=cacheDir,
        chan3_col_name =chan3_col_name,
        label_name=label_name
        #maxSize=maxSize
        ,RandGaussianNoised_prob=experiment.get_parameter("RandGaussianNoised_prob")
        ,RandAdjustContrastd_prob=experiment.get_parameter("RandAdjustContrastd_prob")
        ,RandGaussianSmoothd_prob=experiment.get_parameter("RandGaussianSmoothd_prob")
        ,RandRicianNoised_prob=experiment.get_parameter("RandRicianNoised_prob")
        ,RandFlipd_prob=experiment.get_parameter("RandFlipd_prob")
        ,RandAffined_prob=experiment.get_parameter("RandAffined_prob")
        ,RandCoarseDropoutd_prob=experiment.get_parameter("RandCoarseDropoutd_prob")
        ,is_whole_to_train= (sizeWord=="_maxSize_")
    )
    data.prepare_data()
    data.setup()
    # definition described in model folder
    # from https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/unet/training_setup/neural_networks/unets.py
    unet= unets.UNet(
        spatial_dims=3,
        in_channels=3,
        out_channels=1,
        strides=getParam(experiment,options,"stridesAndChannels",df)["strides"],
        channels=getParam(experiment,options,"stridesAndChannels",df)["channels"],
        num_res_units= experiment.get_parameter("num_res_units"),
        act = getParam(experiment,options,"act",df),
        norm= getParam(experiment,options,"norm",df),
        dropout= experiment.get_parameter("dropout")
    )
    model = LigtningModel.Model(
        net=unet,
        criterion=  getParam(experiment,options,"lossF",df),# Our seg labels are single channel images indicating class index, rather than one-hot
        learning_rate=1e-2,
        optimizer_class= getParam(experiment,options,"optimizer_class",df) ,
        experiment=experiment,
        finalLoss=finalLoss
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
    )
    #stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging()
    trainer = pl.Trainer(
        #accelerator="cpu", #TODO(remove)
        max_epochs=experiment.get_parameter("max_epochs"),
        #gpus=1,
        precision=experiment.get_parameter("precision"), 
        callbacks=[ early_stopping ],
        #logger=comet_logger,
        accelerator='auto',
        devices='auto',       
        default_root_dir= "/home/sliceruser/data/lightning_logs",
        auto_scale_batch_size="binsearch",
        auto_lr_find=True,
        check_val_every_n_epoch=10,
        accumulate_grad_batches=experiment.get_parameter("accumulate_grad_batches"),
        gradient_clip_val=experiment.get_parameter("gradient_clip_val")# 0.5,2.0
    )
    trainer.logger._default_hp_metric = False
    start = datetime.now()
    print('Training started at', start)
    trainer.fit(model=model, datamodule=data)
    print('Training duration:', datetime.now() - start)
    experiment.log_metric("last_val_loss",finalLoss[0])
    #experiment.log_parameters(parameters)  
    experiment.end()
    # #evaluating on test dataset
    # with torch.no_grad():   
    # for batch in data.test_dataloader():
    #     inputs = batch['image'][tio.DATA].to(device)
    #     labels = model.net(inputs).argmax(dim=1, keepdim=True).cpu()
    #     for i in range(len(inputs)):
    #         break
    #     break   


#experiment.end()
