### Define Data Handling

from comet_ml import Experiment
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
from picai_eval import evaluate
from statistics import mean


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
    def __init__(self, net, criterion, learning_rate, optimizer_class,experiment,finalLoss):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.experiment=experiment
        self.finalLoss=finalLoss
        self.picaiLossArr=[]
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
    
    def prepare_batch(self, batch):
        return batch['chan3_col_name'], batch['label']
    
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
        images, labels = batch['chan3_col_name'], batch["label"]
        #print(f" in validation images {images} labels {labels} "  )

        y_hat = sliding_window_inference(images, (32,32,32), 1, self.net)
        #print(f"sss y_hat {y_hat.size()} labels {labels.size()} labels type {type(labels)} y_hat type {type(y_hat)}   ")

        #labels= torch.nn.functional.one_hot(labels, num_classes=2) 
        y_hat = [self.post_pred(i) for i in decollate_batch(y_hat)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        loss = self.criterion(y_hat, labels)

        metrics = evaluate(
            y_det=y_hat.cpu().detach().numpy(),
            y_true=labels.cpu().detach().numpy(),
        )
        self.picaiLossArr.append(metrics)
        #self.dice_metric(y_pred=y_hat, y=labels)
        # print(f"losss {loss}  ")
        self.log('val_loss', loss)
        return loss

    def validation_epoch_end(self, outputs):
        """
        just in order to log the dice metric on validation data 
        """
        # mean_val_dice = self.dice_metric.aggregate().item()
        # self.dice_metric.reset()
        # if mean_val_dice > self.best_val_dice:
        #     self.best_val_dice = mean_val_dice
        #     self.best_val_epoch = self.current_epoch
        # print(
        #     f"current epoch: {self.current_epoch} "
        #     f"current mean dice: {mean_val_dice:.4f}"
        #     f"\nbest mean dice: {self.best_val_dice:.4f} "
        #     f"at epoch: {self.best_val_epoch}"
        # )

        meanPiecaiMetr= mean(self.picaiLossArr)        
        self.log('val_mean_picai_metr', meanPiecaiMetr)
        self.experiment.log_metric("val_mean_picai_metr_training",meanPiecaiMetr)
        self.finalLoss.append(meanPiecaiMetr)
        return {"log": self.log}

    # def validation_step(self, batch, batch_idx):
    #     y_hat, y = self.infer_batch(batch)
    #     loss = self.criterion(y_hat, y)
    #     self.log('val_loss', loss)
    #     return loss

