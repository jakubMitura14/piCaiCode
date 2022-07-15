### Define Data Handling

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
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset,Dataset,PersistentDataset, pad_list_data_collate, decollate_batch,list_data_collate
from monai.config import print_config
from monai.apps import download_and_extract

sns.set()
plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()

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

import torch.nn as nn
import torch.nn.functional as F

monai.utils.set_determinism()

import importlib.util
import sys

spec = importlib.util.spec_from_file_location("transformsForMain", "/home/sliceruser/data/piCaiCode/preprocessing/transformsForMain.py")
transformsForMain = importlib.util.module_from_spec(spec)
sys.modules["transformsForMain"] = transformsForMain
spec.loader.exec_module(transformsForMain)

spec = importlib.util.spec_from_file_location("ManageMetadata", "/home/sliceruser/data/piCaiCode/preprocessing/ManageMetadata.py")
manageMetaData = importlib.util.module_from_spec(spec)
sys.modules["ManageMetadata"] = manageMetaData
spec.loader.exec_module(manageMetaData)


spec = importlib.util.spec_from_file_location("dataUtils", "/home/sliceruser/data/piCaiCode/dataManag/utils/dataUtils.py")
dataUtils = importlib.util.module_from_spec(spec)
sys.modules["dataUtils"] = dataUtils
spec.loader.exec_module(dataUtils)

# import preprocessing.transformsForMain as transformsForMain
# import preprocessing.ManageMetadata as manageMetaData
# import dataManag.utils.dataUtils as dataUtils
# import multiprocessing



class PiCaiDataModule(pl.LightningDataModule):
    def __init__(self,trainSizePercent,batch_size,num_workers
    ,drop_last,df,cache_dir,chan3_col_name
    ,label_name,
    RandGaussianNoised_prob
    ,RandAdjustContrastd_prob
    ,RandGaussianSmoothd_prob
    ,RandRicianNoised_prob
    ,RandFlipd_prob
    ,RandAffined_prob
    ,RandCoarseDropoutd_prob
    ,is_whole_to_train ):
        super().__init__()
        self.cache_dir=cache_dir
        self.batch_size = batch_size
        self.df = df
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.train_set = None
        self.val_set = None
        self.test_set = None  
        self.trainSizePercent =trainSizePercent
        self.train_files = None
        self.val_files= None
        self.test_files= None
        self.train_ds = None
        self.val_ds= None
        self.test_ds= None        
        self.subjects= None
        self.chan3_col_name=chan3_col_name
        self.label_name=label_name
        self.RandGaussianNoised_prob=RandGaussianNoised_prob
        self.RandAdjustContrastd_prob=RandAdjustContrastd_prob
        self.RandGaussianSmoothd_prob=RandGaussianSmoothd_prob
        self.RandRicianNoised_prob=RandRicianNoised_prob
        self.RandFlipd_prob=RandFlipd_prob
        self.RandAffined_prob=RandAffined_prob
        self.RandCoarseDropoutd_prob=RandCoarseDropoutd_prob
        self.is_whole_to_train=is_whole_to_train 



        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    #TODO replace with https://docs.monai.io/en/stable/data.html
    def splitDataSet(self,patList, trainSizePercent,noTestSet):
        """
        test train validation split
        TODO(balance sets)
        """
        totalLen=len(patList)
        train_test_split( patList  )
        numTrain= math.ceil(trainSizePercent*totalLen)
        numTestAndVal=totalLen-numTrain
        numTest=math.ceil(numTestAndVal*0.5)
        numVal= numTestAndVal-numTest

        # valid_set,test_set = torch.utils.data.random_split(test_and_val_set, [math. ceil(0.5), 0.5])
        print('Train data set:', numTrain)
        print('Test data set:',numTest)
        print('Valid data set:', numVal)
        if(noTestSet):
            return torch.utils.data.random_split(patList, [numTrain,numTestAndVal,0])
        else:    
            return torch.utils.data.random_split(patList, [numTrain,numVal,numTest])

    def setup(self, stage=None):
        set_determinism(seed=0)
        self.subjects = list(map(lambda row: manageMetaData.getMonaiSubjectDataFromDataFrame(row[1],self.chan3_col_name,self.label_name)   , list(self.df.iterrows())))
        train_set, valid_set,test_set = self.splitDataSet(self.subjects , self.trainSizePercent,True)
        
        self.train_subjects = train_set
        self.val_subjects = valid_set
        self.test_subjects = test_set
        train_transforms=transformsForMain.get_train_transforms(
            self.RandGaussianNoised_prob
            ,self.RandAdjustContrastd_prob
            ,self.RandGaussianSmoothd_prob
            ,self.RandRicianNoised_prob
            ,self.RandFlipd_prob
            ,self.RandAffined_prob
            ,self.RandCoarseDropoutd_prob
            ,self.is_whole_to_train )
        val_transforms= transformsForMain.get_val_transforms(self.is_whole_to_train )
        #todo - unhash
        # self.train_ds =  PersistentDataset(data=self.train_subjects, transform=train_transforms,cache_dir=self.cache_dir)
        # self.val_ds=     PersistentDataset(data=self.val_subjects, transform=val_transforms,cache_dir=self.cache_dir)
        # self.test_ds=    PersistentDataset(data=self.test_subjects, transform=val_transforms,cache_dir=self.cache_dir)    

        self.train_ds =  Dataset(data=self.train_subjects, transform=train_transforms)
        self.val_ds=     Dataset(data=self.val_subjects, transform=val_transforms)
        #self.test_ds=    Dataset(data=self.test_subjects, transform=val_transforms)
        
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, drop_last=self.drop_last
                          , shuffle=True,num_workers=self.num_workers,collate_fn=list_data_collate)#,collate_fn=list_data_collate

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, drop_last=self.drop_last,num_workers=self.num_workers,collate_fn=list_data_collate)#,collate_fn=pad_list_data_collate

    # def test_dataloader(self):
    #     return DataLoader(self.test_ds, batch_size= 1, drop_last=False,num_workers=self.num_workers,collate_fn=list_data_collate)#num_workers=self.num_workers,