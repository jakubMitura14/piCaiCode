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
from monai.data import CacheDataset,Dataset,PersistentDataset, list_data_collate, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract

sns.set()
plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()
from monai.transforms import (
    EnsureChannelFirstd,
    Orientationd,
    AsDiscrete,
    AddChanneld,
    Spacingd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
)
print('Last run on', time.ctime())

df = pd.read_csv('/home/sliceruser/labels/processedMetaData.csv')

df = df.loc[ df['isAnyMissing'] ==False ]

def getMonaiSubjectDataFromDataFrame(row):
        """
        given row from data frame prepares Subject object from it
        """
        subject= {"adc": row['adc']        
        , "cor":row['cor']
        , "hbv":row['hbv']
        , "sag":row['sag']
        , "t2w":row['t2w']
        , "isAnythingInAnnotated":row['isAnythingInAnnotated']
        , "patient_id":row['patient_id']
        , "study_id":row['study_id']
        , "patient_age":row['patient_age']
        , "psa":row['psa']
        , "psad":row['psad']
        , "prostate_volume":row['prostate_volume']
        , "histopath_type":row['histopath_type']
        , "lesion_GS":row['lesion_GS']
        , "label":row['reSampledPath']
        
        
        }

        return subject
# list(map(lambda row: getSubjectDataFromDataFrame(row[0][1])   , list(df.iterrows())))
#patList=list(map(lambda row: getSubjectDataFromDataFrame(row[1])   , list(df.iterrows())))


cache_dir='/home/sliceruser/preprocess'
class PiCaiDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, df,trainSizePercent):
        super().__init__()
        self.batch_size = batch_size
        self.df = df
        self.trainSizePercent =trainSizePercent
        self.subjects = None
        self.train_subjects = None
        self.val_subjects = None
        self.test_subjects = None
        self.preprocess = None
        self.transform = None
        
        self.train_set = None
        self.val_set = None
        self.test_set = None        

        #adding temporarly simplified dictionaries
        train_files = None
        val_files= None
        test_files= None
        train_ds = None
        val_ds= None
        test_ds= None


    def splitDataSet(self,patList, trainSizePercent):
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
        return torch.utils.data.random_split(patList, [numTrain,numVal,numTest])
        
    def get_max_shape(self, subjects):
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([s.spatial_shape for s in dataset])
        return shapes.max(axis=0)
    
    def prepare_data(self):
        set_determinism(seed=0)
        self.subjects = list(map(lambda row: getMonaiSubjectDataFromDataFrame(row[1])   , list(df.iterrows())))
        train_set, valid_set,test_set = self.splitDataSet(self.subjects , self.trainSizePercent)
        self.train_subjects = train_set
        self.val_subjects = valid_set
        self.test_subjects = test_set
        #adding temporarly simplified dictionaries
        self.train_files = [
            {"image": pat['t2w']  , "label": pat['label']  }
            for pat  in self.train_subjects
        ]
        self.val_files= [{"image": pat['t2w']  , "label": pat['label']  }
            for pat  in self.val_subjects
        ]
        self.test_files= [{"image": pat['t2w']  , "label": pat['label']  }
            for pat  in self.test_subjects
        ]

    
    def get_preprocessing_transform(self):
        """
        preprocessing based on https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7790158/ https://github.com/NIH-MIP/Radiology_Image_Preprocessing_for_Deep_Learning/blob/main/Codes/Main_Preprocessing.py
        denoising - https://github.com/ketanfatania/QMRI-PnP-Recon-POC     ;   https://github.com/AurelienCD/Resampling_Denoising_Deep_Learning_MRI   https://github.com/AurelienCD/Resampling_Denoising_Deep_Learning_MRI
        bias field correction -https://discourse.itk.org/t/n4-bias-field-correction/3972
        registration and superresolotion - https://github.com/gift-surg/NiftyMIC
        rest of registration- https://github.com/SuperElastix/elastix
        standardization - https://github.com/NIH-MIP/Radiology_Image_Preprocessing_for_Deep_Learning/blob/main/Codes/Main_Preprocessing.py
        My:
        resampling -  https://github.com/AurelienCD/Resampling_Denoising_Deep_Learning_MRI
        """
        torchio_transforms = tio.transforms.EnsureShapeMultiple(8, include=["image"]) # for unet
        val_transforms = Compose(
            [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        EnsureTyped(keys=["image", "label"],data_type='tensor', dtype=torch.float),
        torchio_transforms,
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),#TODO(make more refined)
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
            ]
                )
        #TODO add         
        # preprocess = tio.Compose([
        #     tio.RescaleIntensity((-1, 1)),
        #     tio.CropOrPad(self.get_max_shape(self.subjects)),
        #     tio.EnsureShapeMultiple(8),  # for the U-Net
        #     tio.OneHot(),
        # ])


        return val_transforms
    
    # def get_augmentation_transform(self):
    #     #TODO(use augmentations)
    #     # augment = tio.Compose([
    #     #     tio.RandomAffine(),
    #     #     # tio.RandomGamma(p=0.5),
    #     #     # tio.RandomNoise(p=0.5),
    #     #     tio.RandomMotion(p=0.1),
    #     #     tio.RandomBiasField(p=0.25),
    #     # ])
    #     return augment

    def setup(self, stage=None):
        self.preprocess = self.get_preprocessing_transform()
        # augment = self.get_augmentation_transform()
        #TODO(add augmentations) 
        self.train_ds =  PersistentDataset(
            data=self.train_files, transform=self.preprocess,cache_dir=cache_dir   )
        self.val_ds=  PersistentDataset(
            data=self.train_files, transform=self.preprocess,cache_dir=cache_dir)
        self.test_ds=  PersistentDataset(
             data=self.train_files, transform=self.preprocess,cache_dir=cache_dir)

        # self.train_set = tio.SubjectsDataset(self.train_subjects, transform=self.transform)
        # self.val_set = tio.SubjectsDataset(self.val_subjects, transform=self.preprocess)
        # self.test_set = tio.SubjectsDataset(self.test_subjects, transform=self.preprocess)

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size)
		
		
data = PiCaiDataModule(
    df= df,
    batch_size=2,#TODO(batc size determined by lightning)
    trainSizePercent=0.7
)
data.prepare_data()
data.setup()
# print('Training:  ', len(data.train_set))
# print('Validation: ', len(data.val_set))
# print('Test:      ', len(data.test_set))


class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
    
    def prepare_batch(self, batch):
        return batch['image'], batch['label']
    
    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss
unet = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(8, 16, 32, 64),
    strides=(2, 2, 2),
)

model = Model(
    net=unet,
    criterion=monai.losses.DiceCELoss(softmax=True),
    learning_rate=1e-2,
    optimizer_class=torch.optim.AdamW,
)
early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss',
)
trainer = pl.Trainer(
    gpus=1,
    precision=16,
    callbacks=[early_stopping],
)
trainer.logger._default_hp_metric = False		
		
