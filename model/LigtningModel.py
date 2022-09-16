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

def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res

detectSemiSupervised =loadLib("detectSemiSupervised", "/home/sliceruser/data/piCaiCode/model/detectSemiSupervised.py")
torch.autograd.set_detect_anomaly(True)

def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def saveFilesInDir(gold_arr,y_hat_arr, directory, patId):
    """
    saves arrays in given directory and return paths to them
    """
    # gold_im = sitk.GetImageFromArray(gold_arr)
    # y_hat_im = sitk.GetImageFromArray(y_hat_arr)
    # gold_im_path = join(directory, patId+ "_gold.nii.gz" )
    # yHat_im_path = join(directory, patId+ "_hat.nii.gz" )
    
    gold_im_path = join(directory, patId+ "_gold.npy" )
    yHat_im_path = join(directory, patId+ "_hat.npy" )
    np.save(gold_im_path, gold_arr)
    np.save(yHat_im_path, y_hat_arr)



    # writer = sitk.ImageFileWriter()
    # writer.SetFileName(gold_im_path)
    # writer.Execute(gold_im)

    # writer = sitk.ImageFileWriter()
    # writer.SetFileName(yHat_im_path)
    # writer.Execute(y_hat_im)

    return(gold_im_path,yHat_im_path)


def getArrayFromPath(path):
    image1=sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image1)

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
        self.modelRegression = detectSemiSupervised.UNetToRegresion(2)
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.experiment=experiment
        self.picaiLossArr=[]
        self.post_pred = Compose([ AsDiscrete( to_onehot=2)])
        self.picaiLossArr_auroc=[]
        self.picaiLossArr_AP=[]
        self.picaiLossArr_score=[]

        self.picaiLossArr_auroc_final=picaiLossArr_auroc_final
        self.picaiLossArr_AP_final=picaiLossArr_AP_final
        self.picaiLossArr_score_final=picaiLossArr_score_final
        #temporary directory for validation images and their labels
        self.temp_val_dir=tempfile.mkdtemp()
        self.list_gold_val=[]
        self.list_yHat_val=[]
        self.isAnyNan=False

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
    
    def prepare_batch(self, batch):
        return batch['chan3_col_name'], batch['label'], batch['num_lesions_to_retain']
    
    def infer_batch(self, batch):
        x, y, numLesions = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y, numLesions

    def training_step(self, batch, batch_idx):
        # every second iteration we will do the training for segmentation
        y_hat, y , numLesions= self.infer_batch(batch)
        if self.global_step%2==0:
            loss = self.criterion(y_hat, y)
            self.log('train_loss', loss, prog_bar=True)
            return loss
        # in case we have odd iteration we get access only to number of lesions present in the image not where they are (if they are present at all)    
        else:
            regress_res=self.modelRegression(y_hat)
            numLesions=list(map(lambda entry : int(entry), numLesions ))
            numLesions=torch.Tensor(numLesions).to(self.device)
            return F.smooth_l1_loss(regress_res, numLesions )

    # def validation_step(self, batch, batch_idx):
    #     return 0.5

    def validation_step(self, batch, batch_idx):
        images, y_true,numLesions = batch['chan3_col_name_val'], batch["label_name_val"], batch['num_lesions_to_retain']
        print(f" in validation images {images.size()} labels {y_true.size()} "  )

        patIds=batch['patient_id']
        y_det = sliding_window_inference(images, (32,32,32), 1, self.net)
        
        
        regress_res=self.modelRegression(y_det)
        numLesions=list(map(lambda entry : int(entry), numLesions ))
        numLesions=torch.Tensor(numLesions).to(self.device)
        regressLoss=F.smooth_l1_loss(regress_res, numLesions )

        #marking that we had some Nan numbers in the tensor
        if(torch.sum(torch.isnan( y_det))>0):
            self.isAnyNan=True
        
        
        y_det=torch.sigmoid(y_det)
        # print( f"before extract lesion  sum a {torch.sum(y_hat)  } " )

        y_det = decollate_batch(y_det)
        y_true = decollate_batch(y_true)
        patIds = decollate_batch(patIds)
        print(f" y_det 0 {y_det[0].size()} ")
        #Todo check is the order of dimensions as expected by the library

        y_det=[extract_lesion_candidates( x.cpu().detach().numpy()[1,:,:,:])[0] for x in y_det]
        # y_det=[extract_lesion_candidates( torch.permute(x,(2,1,0,3) ).cpu().detach().numpy()[1,:,:,:])[0] for x in y_det]
        y_true=[x.cpu().detach().numpy()[1,:,:,:] for x in y_true]

        regress_res_cpu=regress_res.cpu().detach().numpy()
        for i in range(0,len(y_true)):
            #if regression tell that there are no changes we want it to zero out the final result
            y_det_curr=y_det[i]
            if(np.rint(regress_res_cpu[i])==0):
                y_det_curr=np.zeros_like(y_det_curr)
            tupl=saveFilesInDir(y_true[i],y_det_curr, self.temp_val_dir, patIds[i])
            self.list_gold_val.append(tupl[0])
            self.list_yHat_val.append(tupl[1])
        #now we need to save files in temporary direcory and save outputs to the appripriate lists wit paths
        
        self.log('val_loss', regressLoss)

        return regressLoss




    def validation_epoch_end(self, outputs):
        """
        just in order to log the dice metric on validation data 
        """

        if(len(self.list_yHat_val)>1 and (not self.isAnyNan)):
            chunkLen=8

            valid_metrics = evaluate(y_det=self.list_yHat_val,
                                y_true=self.list_gold_val,
                                #y_true=iter(y_true),
                                #y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
                                )

            meanPiecaiMetr_auroc=valid_metrics.auroc
            meanPiecaiMetr_AP=valid_metrics.AP
            meanPiecaiMetr_score=valid_metrics.score

            print(f"meanPiecaiMetr_auroc {meanPiecaiMetr_auroc} meanPiecaiMetr_AP {meanPiecaiMetr_AP}  meanPiecaiMetr_score {meanPiecaiMetr_score} "  )

            self.log('val_mean_auroc', meanPiecaiMetr_auroc)
            self.log('val_mean_AP', meanPiecaiMetr_AP)
            self.log('val_mean_score', meanPiecaiMetr_score)

            self.experiment.log_metric('val_mean_auroc', meanPiecaiMetr_auroc)
            self.experiment.log_metric('val_mean_AP', meanPiecaiMetr_AP)
            self.experiment.log_metric('val_mean_score', meanPiecaiMetr_score)


            self.picaiLossArr_auroc_final.append(meanPiecaiMetr_auroc)
            self.picaiLossArr_AP_final.append(meanPiecaiMetr_AP)
            self.picaiLossArr_score_final.append(meanPiecaiMetr_score)

            #resetting to 0 
            self.picaiLossArr_auroc=[]
            self.picaiLossArr_AP=[]
            self.picaiLossArr_score=[]


            #clearing and recreatin temporary directory
            shutil.rmtree(self.temp_val_dir)    
            self.temp_val_dir=tempfile.mkdtemp()
            self.list_gold_val=[]
            self.list_yHat_val=[]
        #in case we have Nan values training is unstable and we want to terminate it     
        if(self.isAnyNan):
            self.log('val_mean_score', -0.2)
            self.picaiLossArr_score_final=[-0.2]
            self.picaiLossArr_AP_final=[-0.2]
            self.picaiLossArr_auroc_final=[-0.2]
            print(" naans in outputt  ")

        #self.isAnyNan=False
        return {"log": self.log}

    # def validation_step(self, batch, batch_idx):
    #     y_hat, y = self.infer_batch(batch)
    #     loss = self.criterion(y_hat, y)
    #     self.log('val_loss', loss)
    #     return loss

