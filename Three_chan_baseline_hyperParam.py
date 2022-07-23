
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
Three_chan_baseline =loadLib("Three_chan_baseline", "/home/sliceruser/data/piCaiCode/Three_chan_baseline.py")


#dirs=[]
#def getPossibleColNames(spacing_keyword,sizeWord ):
for spacing_keyword in ["_med_spac", "_one_spac","_one_and_half_spac", "_two_spac" ]:     
    for sizeWord in ["_maxSize_","_div32_" ]: 
        cacheDir =  "/home/sliceruser/preprocess/monai_persistent_Dataset/{spacing_keyword}/{sizeWord}"
        #creating directory if not yet present
        os.makedirs(cacheDir, exist_ok = True)
        # dirs.append(
        # {f"cache_dir":cacheDir ,
        # "chan3_col_name": f"t2w{spacing_keyword}_3Chan{sizeWord}" 
        # ,"label_name":"label{spacing_keyword}{sizeWord}" 
        # ,"metDataDir":"/home/sliceruser/data/metadata/processedMetaData_current.csv"
        # }
        #)




##options
to_onehot_y_loss= False
options={
"lossF":[monai.losses.FocalLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
        ,monai.losses.DiceLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
        ,monai.losses.DiceFocalLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
],
"stridesAndChannels":  [ {
                                                            "strides":[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
                                                            ,"channels":[32, 64, 128, 256, 512, 1024]
                                                            },
                                                            {
                                                            "strides":[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
                                                            ,"channels":[32, 64, 128, 256, 512]
                                                            }  ],
"optimizer_class": [torch.optim.AdamW, torch.optim.NAdam] ,
"act":[(Act.PRELU, {"init": 0.2}),(Act.LEAKYRELU, {})],                                         
"norm":[(Norm.INSTANCE, {}),(Norm.BATCH, {}) ],
}


#####hyperparameters
# based on https://www.comet.com/docs/python-sdk/introduction-optimizer/
# We only need to specify the algorithm and hyperparameters to use:
config = {
    # We pick the Bayes algorithm:
    "algorithm": "bayes",

    # Declare  hyperparameters in 
    "parameters": {
        "lossF": {"type": "discrete", "values": list(range(0,len(options["lossF"])))},
        "stridesAndChannels": {"type": "discrete", "values":  list(range(0,len(options["stridesAndChannels"])))  },
        "optimizer_class": {"type": "discrete", "values":list(range(0,len(options["optimizer_class"])))  },
        "num_res_units": {"type": "discrete", "values": [0]},
        "act": {"type": "discrete", "values":list(range(0,len(options["act"])))  },#,(Act.LeakyReLU,{"negative_slope":0.1, "inplace":True} )
        "norm": {"type": "discrete", "values": list(range(0,len(options["norm"])))},
        "dropout": {"type": "discrete", "values": [0.0,0.1]},
        "precision": {"type": "discrete", "values": [16]},
        "max_epochs": {"type": "discrete", "values": [300]},#900

        "accumulate_grad_batches": {"type": "discrete", "values": [1,3]},
        "gradient_clip_val": {"type": "discrete", "values": [0.0,0.5,2.0]},

        "RandGaussianNoised_prob": {"type": "float", "min": 0.0, "max": 0.5},
        "RandAdjustContrastd_prob": {"type": "float", "min": 0.0, "max": 0.5},
        "RandGaussianSmoothd_prob": {"type": "float", "min": 0.0, "max": 0.5},
        "RandRicianNoised_prob": {"type": "float", "min": 0.0, "max": 0.5},
        "RandFlipd_prob": {"type": "float", "min": 0.0, "max": 0.5},
        "RandAffined_prob": {"type": "float", "min": 0.0, "max": 0.5},
        "RandCoarseDropoutd_prob":{"type": "float", "min": 0.0, "max": 0.5},
  
        "spacing_keyword": {"type": "categorical", "values": ["_med_spac", "_one_spac","_one_and_half_spac", "_two_spac" ]},#True,False
        "sizeWord": {"type": "categorical", "values": ["_div32_"]},#"_maxSize_" ,"_div32_"
        #"dirs": {"type": "discrete", "values": list(range(0,len(options["dirs"])))},
    },

    # Declare what we will be optimizing, and how:
    "spec": {
    "metric": "last_val_loss_score",
        "objective": "minimize",
    },
}

df = pd.read_csv("/home/sliceruser/data/metadata/processedMetaData_current.csv")
maxSize=manageMetaData.getMaxSize("t2w_med_spac",df)


exampleSpacing="_med_spac"

df= manageMetaData.load_df_only_full(
    df
    ,f"t2w{exampleSpacing}_3Chan_maxSize_"
    ,f"label{exampleSpacing}_maxSize_"
    ,True )
df= manageMetaData.load_df_only_full(
    df
    ,f"t2w{exampleSpacing}_3Chan_div32_"
    ,f"label{exampleSpacing}_div32_"
    ,False )


# Next, create an optimizer, passing in the config:
# (You can leave out API_KEY if you already set it)
#opt = Optimizer(config)


opt = Optimizer(config, api_key="yB0irIjdk9t7gbpTlSUPnXBd4")


# print("zzzzzzzzz")
#  print(opt.get_experiments(
#          api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
#          project_name="picai-hyperparam-search-01"))


for experiment in opt.get_experiments(
        project_name="picai-hyperparam-search-10"):
    print("******* new experiment *****")    
    Three_chan_baseline.mainTrain(experiment,options,df)



# ### pytorch model mainly
# loss=monai.losses.FocalLoss(include_background=False, to_onehot_y=True)
# #loss=monai.losses.metrics.DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
# stridesAndChannels={
# "strides":[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
# ,"channels":[32, 64, 128, 256, 512, 1024]
# }


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

# dirs={
#     "cache_dir":"/home/sliceruser/preprocess/monai_persistent_Dataset"
#     ,"t2w_name":"t2w_med_spac"
#     ,"adc_name":"registered_adc_med_spac"
#     ,"hbv_name":"registered_hbv_med_spac"
#     ,"label_name":"label_med_spac" 
#     ,"metDataDir":"/home/sliceruser/data/metadata/processedMetaData_current.csv"}

#############loading meta data 
# df = pd.read_csv('/home/sliceruser/data/metadata/processedMetaData_current.csv')
# maxSize=manageMetaData.getMaxSize(t2w_name,df)
# df= manageMetaData.load_df_only_full(df,t2w_name,adc_name,hbv_name, label_name,maxSize,is_whole_to_train )


# data = DataModule.PiCaiDataModule(
#     df= df,
#     batch_size=2,#TODO(batc size determined by lightning)
#     trainSizePercent=0.5,# change to 0.7
#     num_workers=os.cpu_count(),
#     drop_last=False,#True,
#     cache_dir=cache_dir,
#     t2w_name=t2w_name,
#     adc_name=adc_name,
#     hbv_name=hbv_name,
#     label_name=label_name,
#     maxSize=maxSize
#     ,RandGaussianNoised_prob=RandGaussianNoised_prob
#     ,RandAdjustContrastd_prob=RandAdjustContrastd_prob
#     ,RandGaussianSmoothd_prob=RandGaussianSmoothd_prob
#     ,RandRicianNoised_prob=RandRicianNoised_prob
#     ,RandFlipd_prob=RandFlipd_prob
#     ,RandAffined_prob=RandAffined_prob
#     ,RandCoarseDropoutd_prob=RandCoarseDropoutd_prob
#     ,is_whole_to_train=is_whole_to_train 
# )
# data.prepare_data()
# data.setup()

# # definition described in model folder
# # from https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/unet/training_setup/neural_networks/unets.py
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

# model = LigtningModel.Model(
#     net=unet,
#     criterion=loss, # Our seg labels are single channel images indicating class index, rather than one-hot
#     learning_rate=1e-2,
#     optimizer_class=optimizer_class,
# )
# early_stopping = pl.callbacks.early_stopping.EarlyStopping(
#     monitor='val_loss',
# )
# trainer = pl.Trainer(
#     #accelerato="cpu", #TODO(remove)
#     max_epochs=max_epochs,
#     #gpus=1,
#     precision=precision, 
#     callbacks=[early_stopping],
#     logger=comet_logger,
#     accelerator='auto',
#     devices='auto',
#     default_root_dir= "/home/sliceruser/lightning_logs",
#     auto_scale_batch_size="binsearch",
#     auto_lr_find=True,
#     stochastic_weight_avg=True,
#     accumulate_grad_batches=accumulate_grad_batches,
#     gradient_clip_val=gradient_clip_val# 0.5,2.0

# )
# trainer.logger._default_hp_metric = False

# start = datetime.now()
# print('Training started at', start)
# trainer.fit(model=model, datamodule=data)
# print('Training duration:', datetime.now() - start)


#experiment.end()
