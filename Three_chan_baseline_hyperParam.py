
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




# Define a Sinkhorn (~Wasserstein) loss between sampled measures
#loss = SamplesLoss(loss="sinkhorn")

##options
to_onehot_y_loss= False
options={
"lossF":[monai.losses.FocalLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
        # ,SamplesLoss(loss="sinkhorn",p=3)
        # ,SamplesLoss(loss="hausdorff",p=3)
        # ,SamplesLoss(loss="energy",p=3)
        #,monai.losses.DiceLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
        #,monai.losses.DiceLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
        #,monai.losses.DiceFocalLoss(include_background=False, to_onehot_y=to_onehot_y_loss)
        
],
"stridesAndChannels":  [ {
                                                            "strides":[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)]
                                                            ,"channels":[32, 64, 128, 256, 512, 1024]
                                                            },
                                                            {
                                                            "strides":[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2)]
                                                            ,"channels":[32, 64, 128, 256, 512]
                                                            }  ],
"optimizer_class": [torch.optim.NAdam,torch.optim.LBFGS] ,# ,torch.optim.LBFGS optim.AggMo,   look in https://pytorch-optimizer.readthedocs.io/en/latest/api.html
"act":[(Act.PRELU, {"init": 0.2})],#,(Act.LEAKYRELU, {})                                         
"norm":[(Norm.INSTANCE, {}),(Norm.BATCH, {}) ],
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
        "stridesAndChannels": {"type": "discrete", "values":  list(range(0,len(options["stridesAndChannels"])))  },
        "optimizer_class": {"type": "discrete", "values":list(range(0,len(options["optimizer_class"])))  },
        "num_res_units": {"type": "discrete", "values": [0,1,2]},
        "act": {"type": "discrete", "values":list(range(0,len(options["act"])))  },#,(Act.LeakyReLU,{"negative_slope":0.1, "inplace":True} )
        "norm": {"type": "discrete", "values": list(range(0,len(options["norm"])))},
        "dropout": {"type": "float", "min": 0.0, "max": 0.5},
        "precision": {"type": "discrete", "values": [16]},
        "max_epochs": {"type": "discrete", "values": [100]},#900

        "accumulate_grad_batches": {"type": "discrete", "values": [1,3,10]},
        "gradient_clip_val": {"type": "discrete", "values": [0.0, 0.2,0.5,2.0,100.0]},#,2.0, 0.2,0.5

        "RandGaussianNoised_prob": {"type": "float", "min": 0.0, "max": 0.5},
        "RandAdjustContrastd_prob": {"type": "float", "min": 0.3, "max": 0.8},
        "RandGaussianSmoothd_prob": {"type": "discrete", "values": [0.0]},
        "RandRicianNoised_prob": {"type": "float", "min": 0.2, "max": 0.7},
        "RandFlipd_prob": {"type": "float", "min": 0.3, "max": 0.7},
        "RandAffined_prob": {"type": "float", "min": 0.0, "max": 0.5},
        "RandCoarseDropoutd_prob":{"type": "discrete", "values": [0.0]},
  
        "spacing_keyword": {"type": "categorical", "values": [ "_one_spac" ]},#"_med_spac","_one_and_half_spac", "_two_spac"
        "sizeWord": {"type": "categorical", "values": ["_div32_"]},#,"_maxSize_"# ,"_div32_"
        #"dirs": {"type": "discrete", "values": list(range(0,len(options["dirs"])))},
    },

    # Declare what we will be optimizing, and how:
    "spec": {
    "metric": "last_val_loss_score",
        "objective": "maximize",
    },
}

df = pd.read_csv("/home/sliceruser/data/metadata/processedMetaData_current.csv")
maxSize=manageMetaData.getMaxSize("t2w_med_spac",df)


exampleSpacing="_med_spac"
t2www=f"t2w{exampleSpacing}_3Chan_maxSize_"
labb=f"label{exampleSpacing}_maxSize_"

df= manageMetaData.load_df_only_full(
    df
    ,t2www
    ,labb
    ,True )
df= manageMetaData.load_df_only_full(
    df
    ,f"t2w{exampleSpacing}_3Chan_div32_"
    ,f"label{exampleSpacing}_div32_"
    ,False )
#COMET INFO: COMET_OPTIMIZER_ID=bfa44ecc70f348f1b05ecefcf8f7cd29

# Next, create an optimizer, passing in the config:
# (You can leave out API_KEY if you already set it)
#opt = Optimizer(config)
# opt = Optimizer("7f2a2ec647dc499086a7affb7578b574", api_key="yB0irIjdk9t7gbpTlSUPnXBd4"
# ,trials=500)
opt = Optimizer(config, api_key="yB0irIjdk9t7gbpTlSUPnXBd4",trials=500)
# print("zzzzzzzzz")
#  print(opt.get_experiments(
#          api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
#          project_name="picai-hyperparam-search-01"))


for experiment in opt.get_experiments(
        project_name="picai-hyperparam-search-26"):
    print("******* new experiment *****")    
    Three_chan_baseline.mainTrain(experiment,options,df)


