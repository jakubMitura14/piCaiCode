
from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
from comet_ml import Optimizer

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

import importlib.util
import sys
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray.tune.schedulers.pb2 import PB2
# Import placement group APIs.
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group
)
from pytorch_lightning.loggers import TensorBoardLogger
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray_lightning.tune import TuneReportCallback, get_tune_resources

# torch.multiprocessing.freeze_support()

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
Three_chan_baseline =loadLib("Three_chan_baseline", "/home/sliceruser/data/piCaiCode/Three_chan_baseline.py")
detectSemiSupervised =loadLib("detectSemiSupervised", "/home/sliceruser/data/piCaiCode/model/detectSemiSupervised.py")
semisuperPreprosess =loadLib("semisuperPreprosess", "/home/sliceruser/data/piCaiCode/preprocessing/semisuperPreprosess.py")

import ray
ray.init(runtime_env={"env_vars": {"PL_DISABLE_FORK": "1"}})
##options




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
"models":[getVNet,getSegResNet],
"regression_channels":[[1,1,1],[2,4,8],[10,16,32]],

# "lossF":[monai.losses.FocalLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
#         # ,SamplesLoss(loss="sinkhorn",p=3)
#         # ,SamplesLoss(loss="hausdorff",p=3)
#         # ,SamplesLoss(loss="energy",p=3)
        
# ],

"optimizer_class": [getOptNAdam] ,#torch.optim.LBFGS ,torch.optim.LBFGS optim.AggMo,   look in https://pytorch-optimizer.readthedocs.io/en/latest/api.html
"act":[(Act.PRELU, {"init": 0.2})],#,(Act.LEAKYRELU, {})                                         
"norm":[(Norm.BATCH, {}) ],
"centerCropSize":[(256, 256,32)],
#TODO() learning rate schedulers https://medium.com/mlearning-ai/make-powerful-deep-learning-models-quickly-using-pytorch-lightning-29f040158ef3
}






df = pd.read_csv("/home/sliceruser/data/metadata/processedMetaData_current_b.csv")
spacings =  ["_one_spac_c" ,"_med_spac_b" ]# ,"_med_spac_b" #config['parameters']['spacing_keyword']["values"]

def getDummy(spac):
    label_name=f"label{spac}_maxSize_" 
    imageRef_path=list(filter(lambda it: it!= '', df[label_name].to_numpy()))[0]
    dummyLabelPath=f"/home/sliceruser/data/dummyData/zeroLabel{spac}.nii.gz"
    sizz=semisuperPreprosess.writeDummyLabels(dummyLabelPath,imageRef_path)
    img_size = sizz#(sizz[2],sizz[1],sizz[0])
    return(dummyLabelPath,img_size)


aa=list(map(getDummy  ,spacings  ))
dummyDict={"_one_spac_c" :aa[0],"_med_spac_b":aa[1]   }


################## TUNE definitions



config = {
    "lr": 1e-3,
        #"lossF":list(range(0,len(options["lossF"])))[0],
        #"regression_channels":  tune.choice( list(range(0,len(options["regression_channels"])))),
        #"optimizer_class":  list(range(0,len(options["optimizer_class"])))[0],
        "models":  tune.choice(list(range(0,len(options["models"])))) ,
        "dropout": 0.2,
        "accumulate_grad_batches":  3,
        "spacing_keyword":  "_one_spac_c" ,#,"_med_spac_b"

        "gradient_clip_val": 10.0 ,#{"type": "discrete", "values": [0.0, 0.2,0.5,2.0,100.0]},#,2.0, 0.2,0.5
        "RandGaussianNoised_prob": 0.01,#{"type": "float", "min": 0.0, "max": 0.5},
        "RandAdjustContrastd_prob": 0.4,#{"type": "float", "min": 0.3, "max": 0.8},
        "RandGaussianSmoothd_prob": 0.01,#{"type": "discrete", "values": [0.0]},
        "RandRicianNoised_prob": 0.4,#{"type": "float", "min": 0.2, "max": 0.7},
        "RandFlipd_prob": 0.4,#{"type": "float", "min": 0.3, "max": 0.7},
        "RandAffined_prob": 0.2,#{"type": "float", "min": 0.0, "max": 0.5},
        "RandCoarseDropoutd_prob": 0.01,# {"type": "discrete", "values": [0.0]},
        "RandomElasticDeformation_prob": 0.1,#{"type": "float", "min": 0.0, "max": 0.3},
        "RandomAnisotropy_prob": 0.1,# {"type": "float", "min": 0.0, "max": 0.3},
        "RandomMotion_prob":  0.1,#{"type": "float", "min": 0.0, "max": 0.3},
        "RandomGhosting_prob": 0.1,# {"type": "float", "min": 0.0, "max": 0.3},
        "RandomSpike_prob": 0.1,# {"type": "float", "min": 0.0, "max": 0.3},
        "RandomBiasField_prob": 0.1,# {"type": "float", "min": 0.0, "max": 0.3},

    
}


    # config = {
    #     "layer_1_size": tune.choice([32, 64, 128]),
    #     "layer_2_size": tune.choice([64, 128, 256]),
    #     "lr": 1e-3,
    #     "batch_size": 64,
    # }

pb2_scheduler = PB2(
        time_attr="training_iteration",
        metric='mean_accuracy',
        mode='max',
        perturbation_interval=10.0,
        hyperparam_bounds={
            "lr": [1e-2, 1e-5],
            "gradient_clip_val": [0.0,100.0] ,#{"type": "discrete", "values": [0.0, 0.2,0.5,2.0,100.0]},#,2.0, 0.2,0.5
            "RandGaussianNoised_prob": [0.0,1.0],#{"type": "float", "min": 0.0, "max": 0.5},
            "RandAdjustContrastd_prob": [0.0,1.0],#{"type": "float", "min": 0.3, "max": 0.8},
            "RandGaussianSmoothd_prob": [0.0,1.0],#{"type": "discrete", "values": [0.0]},
            "RandRicianNoised_prob": [0.0,1.0],#{"type": "float", "min": 0.2, "max": 0.7},
            "RandFlipd_prob":[0.0,1.0],#{"type": "float", "min": 0.3, "max": 0.7},
            "RandAffined_prob": [0.0,1.0],#{"type": "float", "min": 0.0, "max": 0.5},
            "RandCoarseDropoutd_prob": [0.0,1.0],# {"type": "discrete", "values": [0.0]},
            "RandomElasticDeformation_prob":[0.0,1.0],#{"type": "float", "min": 0.0, "max": 0.3},
            "RandomAnisotropy_prob": [0.0,1.0],# {"type": "float", "min": 0.0, "max": 0.3},
            "RandomMotion_prob":  [0.0,1.0],#{"type": "float", "min": 0.0, "max": 0.3},
            "RandomGhosting_prob":[0.0,1.0],# {"type": "float", "min": 0.0, "max": 0.3},
            "RandomSpike_prob": [0.0,1.0],# {"type": "float", "min": 0.0, "max": 0.3},
            "RandomBiasField_prob": [0.0,1.0],# {"type": "float", "min": 0.0, "max": 0.3},
            "dropout": [0.0,0.6],# {"type": "float", "min": 0.0, "max": 0.3},
            #"lossF":list(range(0,len(options["lossF"]))),
            #"regression_channels":   list(range(0,len(options["regression_channels"]))),
            #"optimizer_class":  list(range(0,len(options["optimizer_class"]))),
            #"models":  list(range(0,len(options["models"]))) ,

        })

experiment_name="picai-hyperparam-search-30"
# Three_chan_baseline.mainTrain(options,df,experiment_name,dummyDict)
num_workers=2
cpu_num=11 #per gpu
default_root_dir='/home/sliceruser/data/lightninghj'
checkpoint_dir='/home/sliceruser/data/tuneCheckpoints11'
mainTuneDir='/home/sliceruser/data/mainTuneDir'
os.makedirs(checkpoint_dir,  exist_ok = True) 
# os.makedirs(default_root_dir,  exist_ok = True) 
num_cpus_per_worker=cpu_num
reporter = CLIReporter(
        parameter_columns=["lr"],
        metric_columns=["mean_accuracy"])

scheduler = ASHAScheduler(
    max_t=5,
    grace_period=1,
    reduction_factor=2)


tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(
            Three_chan_baseline.mainTrain,
            df=df,
            experiment_name=experiment_name
            ,dummyDict=dummyDict
            ,num_workers=num_workers
            ,cpu_num=cpu_num
             ,default_root_dir=default_root_dir
             ,checkpoint_dir=checkpoint_dir
             ,options=options
             ,num_cpus_per_worker=num_cpus_per_worker            
            ),

        resources=get_tune_resources(num_workers=num_workers, use_gpu=True,num_cpus_per_worker=num_cpus_per_worker)
    ),
    tune_config=tune.TuneConfig(
        metric="mean_accuracy",
        mode="max",
        #scheduler=pb2_scheduler,
        scheduler=scheduler,
        #num_samples=1#num_workers,
    ),
    run_config=air.RunConfig(
        name=experiment_name,
        # progress_reporter=reporter,
    ),
    param_space=config,
    #reuse_actors=True
)
result_grid = tuner.fit()


# result_grid=tune.run(
#         tune.with_parameters(
#             Three_chan_baseline.mainTrain,
#             df=df,
#             experiment_name=experiment_name
#             ,dummyDict=dummyDict
#             ,num_workers=num_workers
#             ,cpu_num=cpu_num
#              ,default_root_dir=default_root_dir
#              ,checkpoint_dir=checkpoint_dir
#              ,options=options
#              ,num_cpus_per_worker=num_cpus_per_worker            
#             ),
#         # metric="ptl/val_accuracy",
#         # mode="max",
#         config=config,
#         num_samples=2,
#         resources_per_trial=get_tune_resources(num_workers=num_workers, use_gpu=True,num_cpus_per_worker=num_cpus_per_worker),
#         scheduler=pb2_scheduler,
#         name=experiment_name,
#         local_dir=mainTuneDir,
#         log_to_file=True,
#         reporter=reporter
        
#         )
        


## printing as defined here https://docs.ray.io/en/latest/ray-air/tuner.html
num_results = len(result_grid)

# Check if there have been errors
if result_grid.errors:
    print("At least one trial failed.")

# Get the best result
best_result = result_grid.get_best_result()

# And the best checkpoint
best_checkpoint = best_result.checkpoint

# And the best metrics
best_metric = best_result.metrics

# Inspect all results
for result in result_grid:
    if result.error:
        print("The trial had an error:", result.error)
        continue

    print("The trial finished successfully with the metrics:", result.metrics["loss"])


# experiment_name="picai-hyperparam-search-30"
# for experiment in opt.get_experiments(
#         project_name=experiment_name):
#     print("******* new experiment *****")    
#     Three_chan_baseline.mainTrain(experiment,options,df,experiment_name,dummyDict)

# os.remove(dummyLabelPath)   




















# def train_mnist_tune(config, num_epochs=10, num_workerss=0, data_dir="~/data"):
#     data_dir = os.path.expanduser(data_dir)
#     model = LightningMNISTClassifier(config, data_dir)
#     trainer = pl.Trainer(
#         max_epochs=num_epochs,
#         # If fractional GPUs passed in, convert to int.
#         gpus=math.ceil(num_workerss),
#         logger=TensorBoardLogger(
#             save_dir=os.getcwd(), name="", version="."),
#         enable_progress_bar=False,
#         callbacks=[
#             TuneReportCallback(
#                 {
#                     "loss": "ptl/val_loss",
#                     "mean_accuracy": "ptl/val_accuracy"
#                 },
#                 on="validation_end")
#         ])
#     trainer.fit(model)


# def tune_mnist_asha(num_samples=10, num_epochs=10, gpus_per_trial=0, data_dir="~/data"):
#     config = {
#         "layer_1_size": tune.choice([32, 64, 128]),
#         "layer_2_size": tune.choice([64, 128, 256]),
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([32, 64, 128]),
#     }

#     scheduler = ASHAScheduler(
#         max_t=num_epochs,
#         grace_period=1,
#         reduction_factor=2)

#     reporter = CLIReporter(
#         parameter_columns=["layer_1_size", "layer_2_size", "lr", "batch_size"],
#         metric_columns=["loss", "mean_accuracy", "training_iteration"])

#     train_fn_with_parameters = tune.with_parameters(train_mnist_tune,
#                                                     num_epochs=num_epochs,
#                                                     num_workerss=gpus_per_trial,
#                                                     data_dir=data_dir)
#     resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}
    
#     tuner = tune.Tuner(
#         tune.with_resources(
#             train_fn_with_parameters,
#             resources=resources_per_trial
#         ),
#         tune_config=tune.TuneConfig(
#             metric="loss",
#             mode="min",
#             scheduler=scheduler,
#             num_samples=num_samples,
#         ),
#         run_config=air.RunConfig(
#             name="tune_mnist_asha",
#             progress_reporter=reporter,
#         ),
#         param_space=config,
#     )
#     results = tuner.fit()

#     print("Best hyperparameters found were: ", results.get_best_result().config)


#     def train_mnist_tune_checkpoint(config,
#                                 checkpoint_dir=None,
#                                 num_epochs=10,
#                                 num_workerss=0,
#                                 data_dir="~/data"):
#     data_dir = os.path.expanduser(data_dir)
#     kwargs = {
#         "max_epochs": num_epochs,
#         # If fractional GPUs passed in, convert to int.
#         "gpus": math.ceil(num_workerss),
#         "logger": TensorBoardLogger(
#             save_dir=os.getcwd(), name="", version="."),
#         "enable_progress_bar": False,
#         "callbacks": [
#             TuneReportCheckpointCallback(
#                 metrics={
#                     "loss": "ptl/val_loss",
#                     "mean_accuracy": "ptl/val_accuracy"
#                 },
#                 filename="checkpoint",
#                 on="validation_end")
#         ]
#     }

#     if checkpoint_dir:
#         kwargs["resume_from_checkpoint"] = os.path.join(
#             checkpoint_dir, "checkpoint")

#     model = LightningMNISTClassifier(config=config, data_dir=data_dir)
#     trainer = pl.Trainer(**kwargs)

#     trainer.fit(model)