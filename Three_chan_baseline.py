
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
import functools
import multiprocessing as mp
import os
import os.path
monai.utils.set_determinism()
from functools import partial

import importlib.util
import sys

percentSplit=0.8

def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res

manageMetaData =loadLib("ManageMetadata", "/home/sliceruser/data/piCaiCode/preprocessing/ManageMetadata.py")
dataUtils =loadLib("dataUtils", "/home/sliceruser/data/piCaiCode/dataManag/utils/dataUtils.py")

unets =loadLib("unets", "/home/sliceruser/data/piCaiCode/model/unets.py")
DataModule =loadLib("DataModule", "/home/sliceruser/data/piCaiCode/model/DataModule.py")
LigtningModel =loadLib("LigtningModel", "/home/sliceruser/data/piCaiCode/model/LigtningModel.py")



def getParam(experiment,options,key,df):
    """
    given integer returned from experiment 
    it will look into options dictionary and return required object
    """
    integerr=experiment.get_parameter(key)
    # print("keyy {key} ")
    # print(options[key])
    return options[key][integerr]

def isAnnytingInAnnotatedInner(row,colName):
    row=row[1]
    path=row[colName]
    image1 = sitk.ReadImage(path)
    #image1 = sitk.Cast(image1, sitk.sitkFloat32)
    data = sitk.GetArrayFromImage(image1)
    return np.sum(data)


def mainTrain(experiment,options,df):
    picaiLossArr_auroc_final=[]
    picaiLossArr_AP_final=[]
    picaiLossArr_score_final=[]
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
    spacing_keyword="_one_spac_c" 
    sizeWord= "_maxSize_" #config["sizeWord")
    chan3_col_name=f"t2w{spacing_keyword}_3Chan{sizeWord}" 
    chan3_col_name_val=chan3_col_name 
    df=df.loc[df[chan3_col_name] != ' ']
    label_name=f"label{spacing_keyword}{sizeWord}" 
    label_name_val=label_name
    df=df.loc[df[label_name_val] != ' ']

    cacheDir =  f"/home/sliceruser/preprocess/monai_persistent_Dataset/{spacing_keyword}/{sizeWord}"

    ##filtering out some pathological cases
    # resList=[]     
    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(isAnnytingInAnnotatedInner,colName=label_name),list(df.iterrows()))    
    # df['locIsInAnnot']= resList
    # df = df.loc[df['locIsInAnnot']>0]


    data = DataModule.PiCaiDataModule(
        df= df,
        batch_size=20,#
        trainSizePercent=percentSplit,# TODO(change to 0.7 or 0.8
        num_workers=os.cpu_count(),
        drop_last=False,#True,
        #we need to use diffrent cache folders depending on weather we are dividing data or not
        cache_dir=cacheDir,
        chan3_col_name =chan3_col_name,
        chan3_col_name_val=chan3_col_name_val,
        label_name_val=label_name_val,
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
        in_channels=4,
        out_channels=2,
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
        picaiLossArr_auroc_final=picaiLossArr_auroc_final,
        picaiLossArr_AP_final=picaiLossArr_AP_final,
        picaiLossArr_score_final=picaiLossArr_score_final
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_mean_score',
        patience=4,
        mode="max",
        divergence_threshold=(-0.1)
    )
    #stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging()
    trainer = pl.Trainer(
        #accelerator="cpu", #TODO(remove)
        max_epochs=experiment.get_parameter("max_epochs"),
        #gpus=1,
        #precision=experiment.get_parameter("precision"), 
        callbacks=[ early_stopping ],
        #logger=comet_logger,
        accelerator='auto',
        devices='auto',       
        default_root_dir= "/home/sliceruser/data/lightning_logs",
        auto_scale_batch_size="binsearch",
        auto_lr_find=True,
        check_val_every_n_epoch=10,
        accumulate_grad_batches=experiment.get_parameter("accumulate_grad_batches"),
        gradient_clip_val=experiment.get_parameter("gradient_clip_val"),# 0.5,2.0
        log_every_n_steps=30,
        strategy='dp'
    )
    trainer.logger._default_hp_metric = False
    start = datetime.now()
    print('Training started at', start)
    trainer.fit(model=model, datamodule=data)
    print('Training duration:', datetime.now() - start)


    experiment.log_metric("last_val_loss_auroc",np.nanmax(picaiLossArr_auroc_final))
    experiment.log_metric("last_val_loss_Ap",np.nanmax(picaiLossArr_AP_final))
    experiment.log_metric("last_val_loss_score",np.nanmax(picaiLossArr_score_final))

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
