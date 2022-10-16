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
from monai.data import CacheDataset,Dataset,PersistentDataset,LMDBDataset, pad_list_data_collate, decollate_batch,list_data_collate,SmartCacheDataset
from monai.config import print_config
from monai.apps import download_and_extract

from torch.utils.data import DataLoader, BatchSampler, RandomSampler
import random
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


# spec = importlib.util.spec_from_file_location("transformsForMain", "/home/sliceruser/locTemp/piCaiCode/preprocessing/transformsForMain.py")
# transformsForMain = importlib.util.module_from_spec(spec)
# sys.modules["transformsForMain"] = transformsForMain
# spec.loader.exec_module(transformsForMain)

# spec = importlib.util.spec_from_file_location("ManageMetadata", "/home/sliceruser/locTemp/piCaiCode/preprocessing/ManageMetadata.py")
# manageMetaData = importlib.util.module_from_spec(spec)
# sys.modules["ManageMetadata"] = manageMetaData
# spec.loader.exec_module(manageMetaData)

from model import transformsForMain as transformsForMain

# spec = importlib.util.spec_from_file_location("dataUtils", "/home/sliceruser/locTemp/piCaiCode/dataManag/utils/dataUtils.py")
# dataUtils = importlib.util.module_from_spec(spec)
# sys.modules["dataUtils"] = dataUtils
# spec.loader.exec_module(dataUtils)

# import preprocessing.transformsForMain as transformsForMain
# import preprocessing.ManageMetadata as manageMetaData
# import dataManag.utils.dataUtils as dataUtils
# import multiprocessing
from monai.config import KeysCollection
from monai.data import MetaTensor
import torchio
import numpy as np


def getMonaiSubjectDataFromDataFrame(row,label_name,label_name_val,t2wColName
,adcColName,hbvColName ):
        """
        given row from data frame prepares Subject object from it
        """
        subject= {#"chan3_col_name": str(row[chan3_col_name])
        "t2w": str(row[t2wColName]),        
        "t2wb": str(row[t2wColName])        
        ,"hbv": str(row[adcColName])        
        ,"adc": str(row[hbvColName]) 
        ,"labelB"    :str(row[label_name])    
        
       , "isAnythingInAnnotated":int(row['isAnythingInAnnotated'])
        , "study_id":str(row['study_id'])
        , "patient_id":str(row['patient_id'])
        , "num_lesions_to_retain":int(row['num_lesions_to_retain_bin'])
        # , "study_id":row['study_id']
        # , "patient_age":row['patient_age']
        # , "psa":row['psa']
        # , "psad":row['psad']
        # , "prostate_volume":row['prostate_volume']
        # , "histopath_type":row['histopath_type']
        # , "lesion_GS":row['lesion_GS']
        , "label":str(row[label_name])
        , "label_name_val":str(row[label_name_val])
        
        
        }

        return subject




class PiCaiDataModule(pl.LightningDataModule):
    def __init__(self,trainSizePercent,batch_size,num_workers
    ,drop_last,df,chan3_col_name,chan3_col_name_val
    ,label_name,label_name_val,
    t2wColName,adcColName,hbvColName,
    RandAdjustContrastd_prob
    ,RandGaussianSmoothd_prob
    ,RandRicianNoised_prob
    ,RandFlipd_prob
    ,RandAffined_prob
    ,RandomElasticDeformation_prob
    ,RandomAnisotropy_prob
    ,RandomMotion_prob
    ,RandomGhosting_prob
    ,RandomSpike_prob
    ,RandomBiasField_prob
    ,persistent_cache  ):
        super().__init__()
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
        self.chan3_col_name_val=chan3_col_name_val
        self.label_name=label_name
        self.label_name_val=label_name_val
        self.t2wColName=t2wColName
        self.adcColName=adcColName
        self.hbvColName=hbvColName
        self.RandAdjustContrastd_prob=RandAdjustContrastd_prob
        self.RandGaussianSmoothd_prob=RandGaussianSmoothd_prob
        self.RandRicianNoised_prob=RandRicianNoised_prob
        self.RandFlipd_prob=RandFlipd_prob
        self.RandAffined_prob=RandAffined_prob
        self.RandomElasticDeformation_prob=RandomElasticDeformation_prob
        self.RandomAnisotropy_prob=RandomAnisotropy_prob
        self.RandomMotion_prob=RandomMotion_prob
        self.RandomGhosting_prob=RandomGhosting_prob
        self.RandomSpike_prob=RandomSpike_prob
        self.RandomBiasField_prob=RandomBiasField_prob
        self.persistent_cache=persistent_cache

    """
    splitting for test and validation and separately in case of examples with some label inside 
        and ecxamples without such constraint
    """
    def getSubjects(self):
        self.df=self.df.loc[self.df['study_id'] !=1000110]# becouse there is error in this label
        self.df=self.df.loc[self.df['study_id'] !=1001489]# becouse there is error in this label
        #onlyPositve = self.df.loc[self.df['isAnyMissing'] ==False]
        onlyPositve = self.df.loc[self.df['isAnythingInAnnotated']>0 ]

        allSubj=list(map(lambda row: getMonaiSubjectDataFromDataFrame(row[1]
        ,label_name=self.label_name,label_name_val=self.label_name
        ,t2wColName=self.t2wColName
        ,adcColName=self.adcColName,hbvColName=self.hbvColName )   , list(self.df.iterrows())))
        
        onlyPositiveSubj=list(map(lambda row: getMonaiSubjectDataFromDataFrame(row[1]
        ,label_name=self.label_name,label_name_val=self.label_name
        ,t2wColName=self.t2wColName
        ,adcColName=self.adcColName,hbvColName=self.hbvColName )  , list(onlyPositve.iterrows())))
        
        return allSubj,onlyPositiveSubj

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
        # self.subjects = list(map(lambda row: getMonaiSubjectDataFromDataFrame(row[1]
        # ,self.label_name,self.label_name_val
        #     ,self.t2wColName, self.adcColName,self.hbvColName )   , list(self.df.iterrows())))
        # train_set, valid_set,test_set = self.splitDataSet(self.subjects , self.trainSizePercent,True)
        
        #train_subjects=self.subjects[0:179]
        #val_subjects=self.subjects[180:200]
        # train_subjects = train_set
        # val_subjects = valid_set+test_set
        # self.test_subjects = test_set

        allSubj,onlyPositve=  self.getSubjects()
        allSubjects= allSubj
        onlyPositiveSubjects= onlyPositve        
        random.shuffle(allSubjects)
        random.shuffle(onlyPositiveSubjects)


        self.allSubjects= allSubjects
        self.onlyPositiveSubjects=onlyPositiveSubjects
        onlyNegative=list(filter(lambda subj :  subj['num_lesions_to_retain']==0  ,allSubjects))        
        noLabels=list(filter(lambda subj :  subj['isAnythingInAnnotated']==0 and subj['num_lesions_to_retain']==1 ,allSubjects))        
        print(f" onlyPositiveSubjects {len(onlyPositiveSubjects)} onlyNegative {len(onlyNegative)} noLabels but positive {len(noLabels)}  ")

        train_transforms=transformsForMain.get_train_transforms(
            self.RandAdjustContrastd_prob
            ,self.RandGaussianSmoothd_prob
            ,self.RandRicianNoised_prob
            ,self.RandFlipd_prob
            ,self.RandAffined_prob
            ,self.RandomElasticDeformation_prob
            ,self.RandomAnisotropy_prob
            ,self.RandomMotion_prob
            ,self.RandomGhosting_prob
            ,self.RandomSpike_prob
            ,self.RandomBiasField_prob          
             )
        train_transforms_noLabel=transformsForMain.get_train_transforms_noLabel(
            self.RandAdjustContrastd_prob
            ,self.RandGaussianSmoothd_prob
            ,self.RandRicianNoised_prob
            ,self.RandFlipd_prob
            ,self.RandAffined_prob
            ,self.RandomElasticDeformation_prob
            ,self.RandomAnisotropy_prob
            ,self.RandomMotion_prob
            ,self.RandomGhosting_prob
            ,self.RandomSpike_prob
            ,self.RandomBiasField_prob          
             )


        val_transforms= transformsForMain.get_val_transforms()

        # self.val_ds=     Dataset(data=onlyPositiveSubjects[0:25]+onlyNegative[0:10], transform=val_transforms)
        # self.train_ds_labels = Dataset(data=onlyPositiveSubjects[25:]+onlyNegative[10:], transform=train_transforms)

        # self.val_ds=     LMDBDataset(data=onlyPositiveSubjects[0:25]+onlyNegative[0:10], transform=val_transforms ,cache_dir=self.persistent_cache)
        # self.train_ds_labels = LMDBDataset(data=onlyPositiveSubjects[25:]+onlyNegative[10:], transform=train_transforms  ,cache_dir=self.persistent_cache)
                #self.train_ds_no_labels = SmartCacheDataset(data=noLabels, transform=train_transforms  ,num_init_workers=os.cpu_count(),num_replace_workers=os.cpu_count())
        # self.val_ds=     SmartCacheDataset(data=onlyPositiveSubjects[0:25]+onlyNegative[0:10], transform=val_transforms  ,num_init_workers=os.cpu_count(),num_replace_workers=os.cpu_count())
        # self.train_ds_labels = SmartCacheDataset(data=onlyPositiveSubjects[25:]+onlyNegative[10:], transform=train_transforms  ,num_init_workers=os.cpu_count(),num_replace_workers=os.cpu_count())
        # self.train_ds_no_labels = SmartCacheDataset(data=noLabels, transform=train_transforms_noLabel  ,num_init_workers=os.cpu_count(),num_replace_workers=os.cpu_count())

        # self.train_ds_all =  LMDBDataset(data=train_set_all, transform=train_transforms,cache_dir=self.persistent_cache)
        onlyPosTreshold=18
        onlyNegativeThreshold=6
        self.val_ds=  Dataset(data=onlyPositiveSubjects[0:onlyPosTreshold]+onlyNegative[0:onlyNegativeThreshold], transform=val_transforms )
        self.train_ds_labels = Dataset(data=onlyPositiveSubjects[onlyPosTreshold:]+onlyNegative[onlyNegativeThreshold:], transform=train_transforms )
        self.train_ds_no_labels = Dataset(data=noLabels, transform=train_transforms_noLabel)
        # self.val_ds=  Dataset(data=onlyPositiveSubjects[0:onlyPosTreshold]+onlyNegative[0:onlyNegativeThreshold], transform=val_transforms )
        # self.train_ds_labels = Dataset(data=onlyPositiveSubjects[onlyPosTreshold:]+onlyNegative[onlyNegativeThreshold:], transform=train_transforms )
        # self.train_ds_no_labels = Dataset(data=noLabels, transform=train_transforms_noLabel)


    def train_dataloader(self):
        # return DataLoader(self.train_ds_labels, batch_size=self.batch_size, drop_last=self.drop_last
        #                   ,num_workers=self.num_workers, shuffle=False )
        return {'train_ds_labels': DataLoader(self.train_ds_labels, batch_size=self.batch_size, drop_last=self.drop_last
                          ,num_workers=self.num_workers,collate_fn=list_data_collate, shuffle=False ),
                'train_ds_no_labels' : DataLoader(self.train_ds_no_labels, batch_size=self.batch_size, drop_last=self.drop_last
                          ,num_workers=self.num_workers,collate_fn=list_data_collate, shuffle=False)           
                          }# ,collate_fn=list_data_collate ,collate_fn=list_data_collate , shuffle=True ,collate_fn=list_data_collate

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size
        , drop_last=self.drop_last,num_workers=self.num_workers, shuffle=False)#,collate_fn=list_data_collate,collate_fn=pad_list_data_collate

    # def test_dataloader(self):
    #     return DataLoader(self.test_ds, batch_size= 1, drop_last=False,num_workers=self.num_workers,collate_fn=list_data_collate)#num_workers=self.num_workers,
