### Define Data Handling

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
#from picai_eval.picai_eval import evaluate_case
from statistics import mean
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter
import tempfile
import shutil
from os import path as pathOs
from os.path import basename, dirname, exists, isdir, join, split
import torch.nn as nn
import torch.nn.functional as F

from monai.metrics import (DiceMetric, HausdorffDistanceMetric,
                           SurfaceDistanceMetric)
from torchmetrics import Precision
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
import importlib.util
import sys
import warnings
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export
import functools
import operator
from torch.nn.intrinsic.qat import ConvBnReLU3d

import multiprocessing as mp
import time
from functools import partial
from torchmetrics.functional import precision_recall
from torch.utils.cpp_extension import load
import torchmetrics
# lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)

class UNetToRegresion(nn.Module):
    def __init__(self,
        in_channels,
        regression_channels
        ,segmModel
    ) -> None:
        super().__init__()
        self.segmModel=segmModel
        self.model = nn.Sequential(
            ConvBnReLU3d(in_channels=in_channels, out_channels=regression_channels[0], kernel_size=3, stride=2,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            ConvBnReLU3d(in_channels=regression_channels[0], out_channels=regression_channels[1], kernel_size=3, stride=2,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            ConvBnReLU3d(in_channels=regression_channels[1], out_channels=regression_channels[2], kernel_size=3, stride=1,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            ConvBnReLU3d(in_channels=regression_channels[2], out_channels=1, kernel_size=3, stride=2,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            nn.AdaptiveMaxPool3d((8,8,2)),#ensuring such dimension 
            nn.Flatten(),
            #nn.BatchNorm3d(8*8*4),
            nn.Linear(in_features=8*8*2, out_features=100),
            #nn.BatchNorm3d(100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=1)
        )
    def forward(self, x):
        segmMap=self.segmModel(x)
        #print(f"segmMap  {segmMap}")
        return (segmMap,self.model(segmMap))




# torch.autograd.set_detect_anomaly(True)

def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def saveFilesInDir(gold_arr,y_hat_arr, directory, patId):
    """
    saves arrays in given directory and return paths to them
    """
    gold_im_path = join(directory, patId+ "_gold.npy" )
    yHat_im_path = join(directory, patId+ "_hat.npy" )
    np.save(gold_im_path, gold_arr)
    np.save(yHat_im_path, y_hat_arr)

    image = sitk.GetImageFromArray(gold_arr)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(join(directory, patId+ "_gold.nii.gz" ))
    writer.Execute(image)


    image = sitk.GetImageFromArray(y_hat_arr)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(join(directory, patId+ "_hat.nii.gz" ))
    writer.Execute(image)

    return(gold_im_path,yHat_im_path)


def saveToValidate(i,y_det,regress_res_cpu,temp_val_dir,y_true,patIds):
    y_det_curr=y_det[i]
    #TODO unhash
    if(np.rint(regress_res_cpu[i])==0):
        y_det_curr=np.zeros_like(y_det_curr)
    return saveFilesInDir(y_true[i],y_det_curr, temp_val_dir, patIds[i])

def getArrayFromPath(path):
    image1=sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image1)

def extractLesions_my(x):
    return extract_lesion_candidates(x)[0]

class Model(pl.LightningModule):
    def __init__(self
    , net
    , criterion
    , learning_rate
    , optimizer_class
    ,picaiLossArr_auroc_final
    ,picaiLossArr_AP_final
    ,picaiLossArr_score_final
    ,regression_channels
    ,lr
    ):
        super().__init__()
        self.lr = learning_rate
        self.modelRegression = UNetToRegresion(2,regression_channels,net)
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.picaiLossArr=[]
        self.post_pred = Compose([ AsDiscrete( to_onehot=2)])
        self.picaiLossArr_auroc=[]
        self.picaiLossArr_AP=[]
        self.picaiLossArr_score=[]
        self.picaiLossArr_auroc_final=picaiLossArr_auroc_final
        self.picaiLossArr_AP_final=picaiLossArr_AP_final
        self.picaiLossArr_score_final=picaiLossArr_score_final
        #temporary directory for validation images and their labels
        self.temp_val_dir= '/home/sliceruser/data/temp' #tempfile.mkdtemp()
        self.list_gold_val=[]
        self.list_yHat_val=[]
        self.isAnyNan=False
        #os.makedirs('/home/sliceruser/data/temp')
        self.postProcess=monai.transforms.Compose([monai.transforms.ForegroundMask()])#, monai.transforms.KeepLargestConnectedComponent()
        self.postTrue = Compose([EnsureType()])
        self.F1Score = torchmetrics.F1Score()
        self.lr=lr
        #shutil.rmtree(self.temp_val_dir) 

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
    

    
    # def infer_batch_pos(self, batch):
    #     x, y, numLesions = batch["pos"]['chan3_col_name'], batch["pos"]['label'], batch["pos"]['num_lesions_to_retain']
    #     y_hat = self.net(x)
    #     return y_hat, y, numLesions



    # def infer_batch_all(self, batch):
    #     x, numLesions =batch["all"]['chan3_col_name'], batch["all"]['num_lesions_to_retain']
    #     y_hat = self.net(x)
    #     return y_hat, numLesions

    def calcLossHelp(self,isAnythingInAnnotated_list,seg_hat_list, y_true_list,reg_hat_list,numLesions_list ,i):
        if(isAnythingInAnnotated_list[i]>0):
            lossSeg=self.criterion(seg_hat_list[i], y_true_list[i])
            lossReg=F.smooth_l1_loss(reg_hat_list[i],numLesions_list[i])
            return torch.add(lossSeg,lossReg)
        return  F.smooth_l1_loss(reg_hat_list[i],numLesions_list[i]) 

    def calculateLoss(self,isAnythingInAnnotated,seg_hat,y_true,reg_hat,numLesions):
        seg_hat_list = decollate_batch(seg_hat)
        isAnythingInAnnotated_list = decollate_batch(isAnythingInAnnotated)
        y_true_list = decollate_batch(y_true)
        reg_hat_list = decollate_batch(reg_hat)
        numLesions_list = decollate_batch(numLesions)
        toSum= list(map(lambda i:  self.calcLossHelp(self,isAnythingInAnnotated_list,seg_hat_list, y_true_list,reg_hat_list,numLesions_list ,i) , list( range(0,len( y_det)) )))
        return torch.sum(torch.stack(toSum))
        #for i in range(0,len( y_det)):
            # if(isAnythingInAnnotated[i]>0):
            #     lossSeg=self.criterion(seg_hat, y_true)
            #     lossReg=F.smooth_l1_loss(reg_hat,numLesions)



    def training_step(self, batch, batch_idx):
        # every second iteration we will do the training for segmentation
        x, y_true, numLesions,isAnythingInAnnotated = batch['chan3_col_name'], batch['label'], batch['num_lesions_to_retain'], batch['isAnythingInAnnotated']
        seg_hat, reg_hat = self.modelRegression(x)
        return self.calculateLoss(isAnythingInAnnotated,seg_hat,y_true,reg_hat,numLesions)

        # if(isAnythingInAnnotated>0):
        #     lossSeg=self.criterion(seg_hat, y_true)
        #     lossReg=F.smooth_l1_loss(reg_hat,numLesions)
        #     return torch.add(lossSeg,lossReg)
        # return  F.smooth_l1_loss(reg_hat,numLesions)   
        
        # # y_hat, y , numLesions_ab= self.infer_batch_pos(batch)
        # lossa = self.criterion(y_hat, y)
        # # regressab=  self.modelRegression(y_hat)
        # # numLesions_ab2=list(map(lambda entry : int(entry), numLesions_ab ))
        # # numLesions_ab3=torch.Tensor(numLesions_ab2).to(self.device)  
        # # lossab=F.smooth_l1_loss(torch.flatten(regressab), torch.flatten(numLesions_ab3) )
      
        # # # in case we have odd iteration we get access only to number of lesions present in the image not where they are (if they are present at all)    
        # # y_hat_all, numLesions= self.infer_batch_all(batch)
        # # regress_res=self.modelRegression(y_hat_all)
        # # numLesions1=list(map(lambda entry : int(entry), numLesions ))
        # # numLesions2=torch.Tensor(numLesions1).to(self.device)
        # # # print(f" regress res {torch.flatten(regress_res).size()}  orig {torch.flatten(numLesions).size() } ")
        # # lossb=F.smooth_l1_loss(torch.flatten(regress_res), torch.flatten(numLesions2) )

        # # self.log('train_loss', torch.add(lossa,lossb), prog_bar=True)
        # # self.log('train_image_loss', lossa, prog_bar=True)
        # # self.log('train_reg_loss', lossb, prog_bar=True)
        # # return torch.add(torch.add(lossa,lossb),lossab)
        # return lossa
    # def validation_step(self, batch, batch_idx):

    def validation_step(self, batch, batch_idx):
        x, y_true, numLesions,isAnythingInAnnotated = batch['chan3_col_name_val'], batch['label_name_val'], batch['num_lesions_to_retain'], batch['isAnythingInAnnotated']
        #print(f"validation_step x {x}  batch['chan3_col_name'] {batch['chan3_col_name']}")
        
        seg_hat, reg_hat = self.modelRegression(x)
        
        val_losss= self.calculateLoss(isAnythingInAnnotated,seg_hat,y_true,reg_hat,numLesions)


        y_det = decollate_batch(seg_hat)
        y_true = decollate_batch(y_true)
        #TODO probably this [1,:,:,:] could break the evaluation ...
        # y_det=[x.cpu().detach().numpy()[1,:,:,:][0] for x in y_det]
        # y_true=[x.cpu().detach().numpy() for x in y_true]
        # y_det= list(map(self.postProcess  , y_det))
        # y_true= list(map(self.postTrue , y_det))

        dice = DiceMetric()
        for i in range(0,len( y_det)):
            if(isAnythingInAnnotated[i]>0):
                #print(f"torch.flatten(regress_res)[i] {torch.flatten(regress_res)[i]}")
                # regress_res_round= round(torch.flatten(regress_res)[i].item())
                #print(f"pre  y_det[i] {y_det[i].size()} y_true_i {y_true[i].size()} ")
                y_det_i=self.postProcess(y_det[i])[0,:,:,:].cpu()
                y_true_i=self.postTrue(y_true[i])[1,:,:,:].cpu()
                #print(f"post  y_det[i] {y_det_i.size()} y_true_i {y_true_i.size()} ")
                if(torch.sum(y_det_i).item()>0 and torch.sum(y_true_i).item()>0 ):
                    # total_loss+= monai.metrics.compute_generalized_dice(y_det_i,y_true_i)/len( y_det)
                    #print(f" monai.metrics.compute_generalized_dice(y_det_i,y_true_i)/len( y_det) {monai.metrics.compute_generalized_dice(y_det_i,y_true_i)/len( y_det)} ")
                    dice(y_det_i,y_true_i)
                    #sd(y_pred=y_det_i, y=y_true_i) 
                # print(f"numLesions[i] {numLesions[i]}")    
                # total_loss+= (abs(regress_res_round-int(numLesions[i]) ) /len( y_det) )#arbitrary number
                
        regress_res2= torch.flatten(reg_hat) 
        regress_res3=list(map(lambda el:round(el) ,torch.flatten(regress_res2).cpu().detach().numpy() ))

        # #print( f"torch.Tensor(numLesions).cpu() {torch.Tensor(numLesions).cpu()}  torch.Tensor(regress_res).cpu() {torch.Tensor(regress_res).cpu()}   ")
        # #self.F1Score(torch.Tensor(regress_res3).int(), torch.Tensor(numLesions2).cpu().int())
        total_loss=precision_recall(torch.Tensor(regress_res3).int(), torch.Tensor(numLesions).cpu().int(), average='macro', num_classes=4)
        total_loss1=torch.mean(torch.stack([total_loss[0],total_loss[1]] ))#self.F1Score
        print(f" total loss a {total_loss1} val_loss {val_losss}")
        total_loss2= torch.add(total_loss1,dice.aggregate())
        print(f" total loss b {total_loss2}  total_loss,dice.aggregate() {dice.aggregate()}")

        # #print(f"sd.aggregate() {sd.aggregate().item()}")
        
        # self.picaiLossArr_score_final.append(total_loss2.item())
        # print(f" validation_acc {total_loss2.item()}  ")
        # self.log("validation_acc", total_loss2.item(), on_epoch=True, on_step=False, sync_dist=True, prog_bar=True, logger=True)
        # self.log("val_loss", val_losss.item(), on_epoch=True, on_step=False, sync_dist=True, prog_bar=True, logger=True)
       
        return {'val_acc': total_loss2.item(), 'val_loss':val_losss}
        #return {'val_acc': dice.aggregate().item(), 'val_loss':val_losss}


    def validation_epoch_end(self, outputs):
        
        avg_loss = torch.mean(torch.stack([torch.as_tensor(x['val_loss']) for x in outputs]))
        print(f"mean_val_loss { avg_loss}")
        avg_acc = torch.mean(torch.stack([torch.as_tensor(x['val_acc']) for x in outputs]))

        self.log("mean_val_loss", avg_loss)
        self.log("mean_val_acc", avg_acc)

        # self.log('ptl/val_loss', avg_loss)
        # self.log('ptl/val_accuracy', avg_acc)
        #return {'mean_val_loss': avg_loss, 'mean_val_acc':avg_acc}



