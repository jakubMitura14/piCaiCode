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
from monai.data import CacheDataset,Dataset,PersistentDataset, list_data_collate, decollate_batch
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

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
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

class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
    
    def prepare_batch(self, batch):
        return batch['common_3channels'], batch['label']
    
    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        #print(f"batch len in training {len(batch)}")

        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch['common_3channels'], batch["label"]
        y_hat = sliding_window_inference(
        images, (32,32,32), 1, self.net)
        loss = self.criterion(y_hat, labels)
        self.log('val_loss', loss)
        return loss


    # def validation_step(self, batch, batch_idx):
    #     y_hat, y = self.infer_batch(batch)
    #     loss = self.criterion(y_hat, y)
    #     self.log('val_loss', loss)
    #     return loss

# class Model(pl.LightningModule):
#     def __init__(self, net, criterion, learning_rate, optimizer_class):
#         super().__init__()
#         self.lr = learning_rate
#         self.net = net
#         self.criterion = criterion
#         self.optimizer_class = optimizer_class
    
#     def configure_optimizers(self):
#         optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
#         return optimizer
    
#     def prepare_batch(self, batch):
#         return batch['t2w'], batch['label']
    
#     def infer_batch(self, batch):
#         x, y = self.prepare_batch(batch)
#         y_hat = self.net(x)
#         return y_hat, y

#     # def training_step(self, batch, batch_idx):
#     #     y_hat, y = self.infer_batch(batch)
#     #     loss = self.criterion(y_hat, y)
#     #     self.log('train_loss', loss, prog_bar=True)
#     #     return loss
    
#     def validation_step(self, batch, batch_idx):
#         y_hat, y = self.infer_batch(batch)
#         loss = self.criterion(y_hat, y)
#         self.log('val_loss', loss)
#         return loss
    
    
#     def training_step(self, batch, batch_idx):

#         #print(f"in training {len(batch)}")
        
#         images, labels = batch["t2w"], batch["label"]
#         output = self.net(images)
#         loss = self.criterion(output, labels)
        
#         # y_hat, y = self.infer_batch(batch)
#         # loss = self.criterion(y_hat, y)
#         self.log('train_loss', loss, prog_bar=True)
#         return loss
    
    # def validation_step(self, batch, batch_idx):        
    #     print(f"in val {len(batch)}")      
    #     print(f"in val image {(batch['common_3channels'].size() )}") 
    #     print(f"in val label {(batch['label'].size() )}") 
        
    #     images, labels = batch['common_3channels'], batch["label"]
    #     outputs = sliding_window_inference(
    #         images, (32,32,32), 1, self.net)
    #     # print(f"in val sliding outputs {outputs.size()}") 
    #     # outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
    #     # print(f"in val sliding outputs post pred {type(outputs)}") 


    #     # print("beeefore model ")
    #     # output = self.net(images)
    #     # print(f"in val output {output.size()}") 


    #     # outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
    #     # labels = [self.post_label(i) for i in decollate_batch(labels)]
    #     # print(f"in val sliding outputs post pred {type(np.array(outputs))}") 
    #     # print(f"in val sliding labels post pred {type(np.array(labels))}") 

    #     print(f"in val sliding outputs post pred {(outputs.size())}") 
    #     print(f"in val sliding labels post pred {(labels.size())}") 


    #     loss = self.criterion(outputs,labels)

    #     #loss = self.criterion(output, labels)
        
    #     return loss

        
        
# #         return loss


# unet = monai.networks.nets.UNet(
#     spatial_dims=3,
#     in_channels=1,
#     out_channels=1,
#     channels=(16, 32, 64, 128, 256),
#     #channels=(16, 16, 16, 16, 16),
#     #strides=(1, 1, 1, 1),
#     strides=(2, 2, 2, 2),

#     num_res_units=1,
# )