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
from report_guided_annotation import extract_lesion_candidates


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
import torchio
def getMeanIgnoreNan(a):
    b=list(filter(lambda it: not np.isnan(it),a))
    # b = a[np.logical_not(np.isnan(a))]
    return np.mean(b)
    
class Model(pl.LightningModule):
    def __init__(self
    , net
    , criterion
    , learning_rate
    , optimizer_class
    ,experiment
    ,picaiLossArr_auroc_final
    ,picaiLossArr_AP_final
    ,picaiLossArr_score_final
    ):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.experiment=experiment
        self.picaiLossArr=[]
        self.post_pred = Compose([ AsDiscrete( to_onehot=2)])
#        self.post_pred = Compose([ AsDiscrete(argmax=True, to_onehot=2)])

        #self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        #self.post_label = Compose([EnsureType("tensor", device="cpu"), torchio.transforms.OneHot(include=["label"] ,num_classes=2)])
        #self.post_label = Compose([EnsureType("tensor", device="cpu"), torchio.transforms.OneHot(include=["label"] ,num_classes=2)])
        self.picaiLossArr_auroc=[]
        self.picaiLossArr_AP=[]
        self.picaiLossArr_score=[]

        self.picaiLossArr_auroc_final=picaiLossArr_auroc_final
        self.picaiLossArr_AP_final=picaiLossArr_AP_final
        self.picaiLossArr_score_final=picaiLossArr_score_final

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
        #print(f"aa  y_hat {y_hat.size()}  y  {y.size()}")  
        #y = torch.stack([self.post_pred(i) for i in decollate_batch(y)])
        #concatLabels= torch.stack(labelsb)
        #print(f"bb  cat y  {y.size()}")  
        
        #print(f"labels {labelsb[0].size()}  labels type {type(labelsb[0])} concatLabels {  concatLabels.size()}  ")
        loss = self.criterion(y_hat, y)

        self.log('train_loss', loss, prog_bar=True)
        return loss
    # def validation_step(self, batch, batch_idx):
    #     return 0.5

    def validation_step(self, batch, batch_idx):
        images, labels = batch['chan3_col_name'], batch["label"]
        #print(f" in validation images {images} labels {labels} "  )

        y_hat = sliding_window_inference(images, (32,32,32), 1, self.net)
        #print(f"sss y_hat {y_hat.size()} labels {labels.size()} labels type {type(labels)} y_hat type {type(y_hat)}   ")
        #print(f"sss a y_hat {y_hat.size()} labels {labels.size()} labels type {type(labels)} y_hat type {type(y_hat)}   ")
        # labelsb = [self.post_pred(i) for i in decollate_batch(labels)]
        # concatLabels= torch.stack(labelsb)
        #print(f"labels {labelsb[0].size()}  labels type {type(labelsb[0])} concatLabels {  concatLabels.size()}  ")
        #print(f" uniqqqque y_hat {torch.unique(y_hat)} y  {torch.unique(labels)}  ")
        loss = self.criterion(y_hat, labels)

        #labels= torch.nn.functional.one_hot(labels, num_classes=2) 
#        y_hat = [(i) for i in decollate_batch(y_hat)]
        y_hat = decollate_batch(y_hat)

        #print(f"sss b  labels type {type(labels)} y_hat type {type(y_hat)}   ")
        #print(f"sss b y_hat {y_hat.size()} labels {labels.size()} labels type {type(labels)} y_hat type {type(y_hat)}   ")
        
        #labelsb = [torch.nn.functional.one_hot(i.to(torch.int64), num_classes=- 1) for i in decollate_batch(labels)]
        #print(f"sss c  labels type {type(labels)} y_hat type {type(y_hat)}   ")

        #print(f"sss c y_hat {y_hat.size()} labels {labels.size()} labels type {type(labels)} y_hat type {type(y_hat)}   ")
        #print(f"sss d y_hat {y_hat[0].size()} labels {labels[0].size()}  labels type {type(labels[0])} y_hat type {type(y_hat[0])}   ")

        #print(f" labels sum {torch.sum(labels)} ")

        valid_metrics = evaluate(y_det=iter(np.concatenate([x.cpu().detach().numpy() for x in y_hat], axis=0)),
                             y_true=iter(np.concatenate([x.cpu().detach().numpy() for x in labels], axis=0)),
                             y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])

        # for i in range(0, len(labelsb)):
        #     metrics = evaluate(
        #         y_det=y_hat[i].cpu().detach().numpy(),
        #         y_true=labelsb[i].cpu().detach().numpy(),
        #     )
        #self.picaiLossArr_auroc.append(valid_metrics.auroc)
        #self.picaiLossArr_AP.append(valid_metrics.AP  )
        self.picaiLossArr_score.append(valid_metrics.score)
        
        
        # meanPiecaiMetr_auroc= mean(self.picaiLossArr_auroc)
        # meanPiecaiMetr_AP= mean(self.picaiLossArr_AP)        
        # meanPiecaiMetr_score= mean(self.picaiLossArr_score)  

        #meanPiecaiMetr_auroc= valid_metrics.auroc
        #meanPiecaiMetr_AP= valid_metrics.AP   
        meanPiecaiMetr_score= valid_metrics.score
        
        
        #print( f"metrics.auroc {meanPiecaiMetr_auroc} metrics.AP {meanPiecaiMetr_AP}  metrics.score {meanPiecaiMetr_score}  " )
        print( f" metrics.score {meanPiecaiMetr_score}  " )

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

        
        # meanPiecaiMetr_auroc= getMeanIgnoreNan(self.picaiLossArr_auroc) # mean(self.picaiLossArr_auroc)
        # meanPiecaiMetr_AP= getMeanIgnoreNan(self.picaiLossArr_AP) # mean(self.picaiLossArr_AP)        
        meanPiecaiMetr_score= getMeanIgnoreNan(self.picaiLossArr_score) #mean(self.picaiLossArr_score)        

        # self.log('val_mean_auroc', meanPiecaiMetr_auroc)
        # self.log('val_mean_AP', meanPiecaiMetr_AP)
        self.log('val_mean_score', meanPiecaiMetr_score)

        # self.experiment.log_metric('val_mean_auroc', meanPiecaiMetr_auroc)
        # self.experiment.log_metric('val_mean_AP', meanPiecaiMetr_AP)
        self.experiment.log_metric('val_mean_score', meanPiecaiMetr_score)


        # self.picaiLossArr_auroc_final.append(meanPiecaiMetr_auroc)
        # self.picaiLossArr_AP_final.append(meanPiecaiMetr_AP)
        self.picaiLossArr_score_final.append(meanPiecaiMetr_score)

        #resetting to 0 
        self.picaiLossArr_auroc=[]
        self.picaiLossArr_AP=[]
        self.picaiLossArr_score=[]

        return {"log": self.log}

    # def validation_step(self, batch, batch_idx):
    #     y_hat, y = self.infer_batch(batch)
    #     loss = self.criterion(y_hat, y)
    #     self.log('val_loss', loss)
    #     return loss

