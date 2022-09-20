
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


torch.multiprocessing.freeze_support()

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



#dirs=[]
#def getPossibleColNames(spacing_keyword,sizeWord ):
# for spacing_keyword in ["_med_spac", "_one_spac","_one_and_half_spac", "_two_spac" ]:     
#     for sizeWord in ["_maxSize_","_div32_" ]: 
#         cacheDir =  "/home/r/preprocess/monai_persistent_Dataset/{spacing_keyword}/{sizeWord}"
#         #creating directory if not yet present
#         os.makedirs(cacheDir, exist_ok = True)




# Define a Sinkhorn (~Wasserstein) loss between sampled measures
#loss = SamplesLoss(loss="sinkhorn")

##options
to_onehot_y_loss= False



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




#getViTAutoEnc,getAhnet,getSegResNetVAE,getAttentionUnet,getSwinUNETR,getSegResNet,getVNet,getUnetB
options={

"models":[getUnetA,getUnetB,getVNet,getSegResNet],
# "models":[getSegResNet],
"regression_channels":[[1,1,1],[2,4,8],[10,16,32]],
#"regression_channels":[[2,4,8]],

"lossF":[monai.losses.FocalLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
        # ,SamplesLoss(loss="sinkhorn",p=3)
        # ,SamplesLoss(loss="hausdorff",p=3)
        # ,SamplesLoss(loss="energy",p=3)
        #,monai.losses.DiceLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
        #,monai.losses.DiceLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
        #,monai.losses.DiceFocalLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
        
],
# "stridesAndChannels":  [ {
#                                                             "strides":[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
#                                                             ,"channels":[32, 64, 128, 256, 512, 1024]
#                                                             },
#                                                             #  {
#                                                             # "strides":[(2, 2, 2), (1, 2, 2),(1, 1, 1), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
#                                                             # ,"channels":[32, 64, 128, 256, 512, 1024, 2048]
#                                                             # },
#                                                             {
#                                                             "strides":[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
#                                                             ,"channels":[32, 64, 128, 256, 512]
#                                                             }  ],
"optimizer_class": [torch.optim.NAdam] ,#torch.optim.LBFGS ,torch.optim.LBFGS optim.AggMo,   look in https://pytorch-optimizer.readthedocs.io/en/latest/api.html
"act":[(Act.PRELU, {"init": 0.2})],#,(Act.LEAKYRELU, {})                                         
"norm":[(Norm.BATCH, {}) ],
"centerCropSize":[(256, 256,32)],
#TODO() learning rate schedulers https://medium.com/mlearning-ai/make-powerful-deep-learning-models-quickly-using-pytorch-lightning-29f040158ef3
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
        "regression_channels": {"type": "discrete", "values": list(range(0,len(options["regression_channels"])))},
        #"stridesAndChannels": {"type": "discrete", "values":  list(range(0,len(options["stridesAndChannels"])))  },
        "optimizer_class": {"type": "discrete", "values":list(range(0,len(options["optimizer_class"])))  },
        "models": {"type": "discrete", "values":list(range(0,len(options["models"])))  },
        # "num_res_units": {"type": "discrete", "values": [0]},#,1,2
        # "act": {"type": "discrete", "values":list(range(0,len(options["act"])))  },#,(Act.LeakyReLU,{"negative_slope":0.1, "inplace":True} )
        # "norm": {"type": "discrete", "values": list(range(0,len(options["norm"])))},
        # "centerCropSize": {"type": "discrete", "values": list(range(0,len(options["centerCropSize"])))},
        "dropout": {"type": "float", "min": 0.0, "max": 0.5},
        #"precision": {"type": "discrete", "values": [16]},
        #"max_epochs": {"type": "discrete", "values": [100]},#900
        "accumulate_grad_batches": {"type": "discrete", "values": [1,3,10]},
        "gradient_clip_val": {"type": "discrete", "values": [0.0, 0.2,0.5,2.0,100.0]},#,2.0, 0.2,0.5
        "RandGaussianNoised_prob": {"type": "float", "min": 0.0, "max": 0.5},
        "RandAdjustContrastd_prob": {"type": "float", "min": 0.3, "max": 0.8},
        "RandGaussianSmoothd_prob": {"type": "discrete", "values": [0.0]},
        "RandRicianNoised_prob": {"type": "float", "min": 0.2, "max": 0.7},
        "RandFlipd_prob": {"type": "float", "min": 0.3, "max": 0.7},
        "RandAffined_prob": {"type": "float", "min": 0.0, "max": 0.5},
        "RandCoarseDropoutd_prob":{"type": "discrete", "values": [0.0]},
        "spacing_keyword": {"type": "categorical", "values": ["_one_spac_c","_med_spac_b" ]},#      #"_med_spac","_one_and_half_spac", "_two_spac"
        #"sizeWord": {"type": "categorical", "values": ["_maxSize_"]},#,"_maxSize_"# ,"_div32_"
        #"dirs": {"type": "discrete", "values": list(range(0,len(options["dirs"])))},
        "RandomElasticDeformation_prob": {"type": "float", "min": 0.0, "max": 0.3},
        "RandomAnisotropy_prob": {"type": "float", "min": 0.0, "max": 0.3},
        "RandomMotion_prob": {"type": "float", "min": 0.0, "max": 0.3},
        "RandomGhosting_prob": {"type": "float", "min": 0.0, "max": 0.3},
        "RandomSpike_prob": {"type": "float", "min": 0.0, "max": 0.3},
        "RandomBiasField_prob": {"type": "float", "min": 0.0, "max": 0.3},
  
    },

    # Declare what we will be optimizing, and how:
    "spec": {
    "metric": "last_val_loss_score",
        "objective": "maximize",
    },
    "trials": 500
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







# maxSize=manageMetaData.getMaxSize("t2w_med_spac_b",df)

# exampleSpacing="_med_spac_b"
# exampleSpacingB="_med_spac_b"
# t2www=f"t2w{exampleSpacing}_3Chan_div32_"
# labb=f"label{exampleSpacing}_div32_"

# df= manageMetaData.load_df_only_full(
#     df
#     ,t2www
#     ,labb
#     ,True ,transformsForMain,t2www,labb )
# df= manageMetaData.load_df_only_full(
#     df
#     ,f"t2w{exampleSpacingB}_3Chan_div32_"
#     ,f"label{exampleSpacingB}_div32_"
#     ,False,transformsForMain,t2www,labb )



#COMET INFO: COMET_OPTIMIZER_ID=bfa44ecc70f348f1b05ecefcf8f7cd29

# Next, create an optimizer, passing in the config:
# (You can leave out API_KEY if you already set it)
# opt = Optimizer("b30cda392d1691665aff2222691d720481c60f92", api_key="yB0irIjdk9t7gbpTlSUPnXBd4",trials=500)


opt = Optimizer(config, api_key="yB0irIjdk9t7gbpTlSUPnXBd4",trials=500)
# print("zzzzzzzzz")
#  print(opt.get_experiments(
#          api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
#          project_name="picai-hyperparam-search-01"))

experiment_name="picai-hyperparam-search-30"
for experiment in opt.get_experiments(
        project_name=experiment_name):
    print("******* new experiment *****")    
    Three_chan_baseline.mainTrain(experiment,options,df,experiment_name,dummyDict)

# os.remove(dummyLabelPath)   

