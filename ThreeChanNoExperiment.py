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
from ray_lightning import RayStrategy

# import preprocessing.transformsForMain
# import preprocessing.ManageMetadata
import model.unets as unets
import model.DataModule as DataModule
import model.LigtningModel as LigtningModel
# import preprocessing.semisuperPreprosess
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray_lightning import RayShardedStrategy

def isAnnytingInAnnotatedInner(row,colName):
    row=row[1]
    path=row[colName]
    image1 = sitk.ReadImage(path)
    #image1 = sitk.Cast(image1, sitk.sitkFloat32)
    data = sitk.GetArrayFromImage(image1)
    return np.sum(data)




def train_model(label_name, dummyLabelPath, df,percentSplit,cacheDir
         ,chan3_col_name,chan3_col_name_val,label_name_val
         ,RandGaussianNoised_prob,RandAdjustContrastd_prob,RandGaussianSmoothd_prob,
         RandRicianNoised_prob,RandFlipd_prob, RandAffined_prob,RandCoarseDropoutd_prob
         ,is_whole_to_train,centerCropSize,
         num_res_units,act,norm,dropout
         ,criterion, optimizer_class,max_epochs,accumulate_grad_batches,gradient_clip_val
         ,picaiLossArr_auroc_final,picaiLossArr_AP_final,picaiLossArr_score_final
          ,experiment_name,net    ,RandomElasticDeformation_prob
    ,RandomAnisotropy_prob
    ,RandomMotion_prob
    ,RandomGhosting_prob
    ,RandomSpike_prob
    ,RandomBiasField_prob,regression_channels,num_workers,cpu_num ,default_root_dir,checkpoint_dir,lr,num_cpus_per_worker):        
    #TODO(remove)
    comet_logger = CometLogger(
        api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
        #workspace="OPI", # Optional
        project_name=experiment_name, # Optional
        #experiment_name="baseline" # Optional
    )
    
    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(addDummyLabelPath,labelName=label_name ,dummyLabelPath= dummyLabelPath ) ,list(df.iterrows())) 


    data = DataModule.PiCaiDataModule(
        df= df,
        batch_size=3,#
        trainSizePercent=percentSplit,# 
        num_workers=cpu_num,#os.cpu_count(),
        drop_last=False,#True,
        #we need to use diffrent cache folders depending on weather we are dividing data or not
        cache_dir=cacheDir,
        chan3_col_name =chan3_col_name,
        chan3_col_name_val=chan3_col_name_val,
        label_name_val=label_name_val,
        label_name=label_name
        #maxSize=maxSize
        ,RandGaussianNoised_prob=RandGaussianNoised_prob
        ,RandAdjustContrastd_prob=RandAdjustContrastd_prob
        ,RandGaussianSmoothd_prob=RandGaussianSmoothd_prob
        ,RandRicianNoised_prob=RandRicianNoised_prob
        ,RandFlipd_prob=RandFlipd_prob
        ,RandAffined_prob=RandAffined_prob
        ,RandCoarseDropoutd_prob=RandCoarseDropoutd_prob
        ,is_whole_to_train=is_whole_to_train
        ,centerCropSize=centerCropSize
        ,RandomElasticDeformation_prob=RandomElasticDeformation_prob
        ,RandomAnisotropy_prob=RandomAnisotropy_prob
        ,RandomMotion_prob=RandomMotion_prob
        ,RandomGhosting_prob=RandomGhosting_prob
        ,RandomSpike_prob=RandomSpike_prob
        ,RandomBiasField_prob=RandomBiasField_prob
    )


    data.prepare_data()
    data.setup()
    # definition described in model folder
    # from https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/unet/training_setup/neural_networks/unets.py
    # unet= unets.UNet(
    #     spatial_dims=3,
    #     in_channels=3,
    #     out_channels=2,
    #     strides=strides,
    #     channels=channels,
    #     num_res_units= num_res_units,
    #     act = act,
    #     norm= norm,
    #     dropout= dropout
    # )

    model = LigtningModel.Model(
        net=net,
        criterion=  criterion,# Our seg labels are single channel images indicating class index, rather than one-hot
        learning_rate=1e-2,
        optimizer_class= optimizer_class,
        picaiLossArr_auroc_final=picaiLossArr_auroc_final,
        picaiLossArr_AP_final=picaiLossArr_AP_final,
        picaiLossArr_score_final=picaiLossArr_score_final,
        regression_channels=regression_channels,
        lr=lr
    )
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='avg_val_loss',
        patience=4,
        mode="max",
        divergence_threshold=(-0.1)
    )
    checkPointCallback=TuneReportCheckpointCallback(
        metrics={
            "loss": "avg_val_loss",
            "mean_accuracy": "avg_val_acc"
        },
        filename="checkpointtt",
        on="validation_end")

    # tuneCallBack=TuneReportCallback(
    #     {
    #         "loss": "avg_val_loss",
    #         "mean_accuracy": "avg_val_acc"
    #     },
    #     on="validation_end")

    strategy = RayShardedStrategy(num_workers=num_workers,num_cpus_per_worker=num_cpus_per_worker,  use_gpu=True)#num_cpus_per_worker=1, num_workers
    #strategy = RayStrategy(num_workers=1, num_cpus_per_worker=num_cpus_per_worker, use_gpu=True)


    # trainer = pl.Trainer(
    #     #accelerator="cpu", #TODO(remove)
    #     #max_epochs=max_epochs,
    #     #gpus=1,
    #     #precision=experiment.get_parameter("precision"), 
    #     callbacks=[ checkPointCallback ],# TODO unhash,#early_stopping
    #     logger=comet_logger,
    #     # accelerator='auto',
    #     # devices='auto',       
    #     default_root_dir= default_root_dir,
    #     #auto_scale_batch_size="binsearch",
    #     auto_lr_find=True,
    #     check_val_every_n_epoch=10,
    #     accumulate_grad_batches=accumulate_grad_batches,
    #     gradient_clip_val=gradient_clip_val,# 0.5,2.0
    #     log_every_n_steps=2,
    #     strategy=strategy#'ddp'#'ddp' # for multi gpu training
    # )
    callbacks=[checkPointCallback]
    kwargs = {
        #"accelerator":'auto',
        "max_epochs": max_epochs,
        "callbacks" :callbacks,
        "logger" : comet_logger,
        "default_root_dir" : default_root_dir,
        "auto_lr_find" : False,
        "check_val_every_n_epoch" : 10,
        "accumulate_grad_batches" : accumulate_grad_batches,
        "gradient_clip_val" :gradient_clip_val,
        "log_every_n_steps" :2,
        "strategy" :strategy#'ddp'
        }

    if os.path.exists(os.path.join(checkpoint_dir, "checkpointtt")):
        kwargs["resume_from_checkpoint"] = os.path.join(
            checkpoint_dir, "checkpointtt")

    trainer = pl.Trainer(**kwargs)


    #stochasticAveraging=pl.callbacks.stochastic_weight_avg.StochasticWeightAveraging()
    # trainer = pl.Trainer(
    #     #accelerator="cpu", #TODO(remove)
    #     #max_epochs=max_epochs,
    #     #gpus=1,
    #     #precision=experiment.get_parameter("precision"), 
    #     callbacks=[ checkPointCallback ],# TODO unhash,#early_stopping
    #     logger=comet_logger,
    #     # accelerator='auto',
    #     # devices='auto',       
    #     default_root_dir= "/home/sliceruser/data/lightning_logs",
    #     #auto_scale_batch_size="binsearch",
    #     auto_lr_find=True,
    #     check_val_every_n_epoch=10,
    #     accumulate_grad_batches=accumulate_grad_batches,
    #     gradient_clip_val=gradient_clip_val,# 0.5,2.0
    #     log_every_n_steps=2,
    #     strategy=strategy#'ddp'#'ddp' # for multi gpu training
    # )
    #setting batch size automatically
    #TODO(unhash)
    #trainer.tune(model, datamodule=data)

    trainer.logger._default_hp_metric = False
    start = datetime.now()
    print('Training started at', start)
    trainer.fit(model=model, datamodule=data)
    print('Training duration:', datetime.now() - start)



 