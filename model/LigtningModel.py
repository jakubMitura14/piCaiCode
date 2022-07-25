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

torch.autograd.set_detect_anomaly(True)

def divide_chunks(l, n):
     
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

def saveFilesInDir(gold_arr,y_hat_arr, directory, patId):
    """
    saves arrays in given directory and return paths to them
    """
    gold_im = sitk.GetImageFromArray(gold_arr)
    y_hat_im = sitk.GetImageFromArray(y_hat_arr)
    gold_im_path = join(directory, patId+ "_gold.nii.gz" )
    yHat_im_path = join(directory, patId+ "_hat.nii.gz" )
    
    writer = sitk.ImageFileWriter()
    writer.SetFileName(gold_im_path)
    writer.Execute(gold_im)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(yHat_im_path)
    writer.Execute(y_hat_im)

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
        #temporary directory for validation images and their labels
        self.temp_val_dir=tempfile.mkdtemp()
        self.list_gold_val=[]
        self.list_yHat_val=[]

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
    # def validation_step(self, batch, batch_idx):
    #     return 0.5

    def validation_step(self, batch, batch_idx):
        images, y_true = batch['chan3_col_name_val'], batch["label_name_val"]
        #print(f" in validation images {images} labels {labels} "  )
        patIds=batch['patient_id']
        y_det = sliding_window_inference(images, (32,32,32), 1, self.net)
        loss = self.criterion(y_det, y_true)
        # y_det=torch.sigmoid(y_det)
        # # print( f"before extract lesion  sum a {torch.sum(y_hat)  } " )

        # y_det = decollate_batch(y_det)
        # y_true = decollate_batch(y_true)
        # patIds = decollate_batch(patIds)
        # #print(f"after decollate  y_hat{y_hat[0].size()} labels{labels[0].size()} y_hat len {len(y_hat)} labels len {len(labels)}")
        # y_det=[extract_lesion_candidates( x.cpu().detach().numpy()[1,:,:,:])[0] for x in y_det]
        # y_true=[x.cpu().detach().numpy()[1,:,:,:] for x in y_true]


        # for i in range(0,len(y_true)):
        #     tupl=saveFilesInDir(y_true[i],y_det[i], self.temp_val_dir, patIds[i])
        #     self.list_gold_val.append(tupl[0])
        #     self.list_yHat_val.append(tupl[1])
        # #now we need to save files in temporary direcory and save outputs to the appripriate lists wit paths
        

        self.log('val_loss', loss)

        return loss




    def validation_epoch_end(self, outputs):
        """
        just in order to log the dice metric on validation data 
        """

        if(len(self.list_yHat_val)>1):
            print(f" leen {len(self.list_yHat_val)}")
            chunkLen=8
            chunksNumb=math.floor(len(self.list_yHat_val)/chunkLen)
            valid_metrics = evaluate(y_det=self.list_yHat_val,
                                y_true=self.list_gold_val,
                                #y_true=iter(y_true),
                                #y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
                                )

            meanPiecaiMetr_auroc=valid_metrics.auroc
            meanPiecaiMetr_AP=valid_metrics.AP
            meanPiecaiMetr_score=valid_metrics.score
            # for i in range(0,chunksNumb):
            #     startIndex= i*chunkLen
            #     endIndex=(i+1)*chunkLen
            #     print(f" startIndex {startIndex}  endIndex {endIndex}")
            #     valid_metrics = evaluate(y_det=list(map(getArrayFromPath, self.list_yHat_val[startIndex:endIndex])),
            #                         y_true=list(map(getArrayFromPath, self.list_gold_val[startIndex:endIndex]  )),
            #                         #y_true=iter(y_true),
            #                         #y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
            #                         )
            #     self.picaiLossArr_auroc.append(valid_metrics.auroc)
            #     self.picaiLossArr_AP.append(valid_metrics.AP)
            #     self.picaiLossArr_score.append(valid_metrics.score)
            
            
            # startIndex= chunksNumb*chunkLen
            # endIndex=len(self.list_yHat_val)
            # if endIndex>startIndex:
            #     print(f" startIndex {startIndex}  endIndex {endIndex}")

            #     # and the last part
            #     valid_metrics = evaluate(y_det=list(map(getArrayFromPath, self.list_yHat_val[startIndex:endIndex])),
            #                             y_true=list(map(getArrayFromPath, self.list_gold_val[startIndex:endIndex]  )),
            #                             #y_true=iter(y_true),
            #                             #y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
            #                             )
            #     self.picaiLossArr_auroc.append(valid_metrics.auroc)
            #     self.picaiLossArr_AP.append(valid_metrics.AP)
            #     self.picaiLossArr_score.append(valid_metrics.score)



            #     meanPiecaiMetr_auroc= np.nanmean(self.picaiLossArr_auroc) 
            #     meanPiecaiMetr_AP=np.nanmean(self.picaiLossArr_AP) 
            #     meanPiecaiMetr_score=np.nanmean(self.picaiLossArr_score) 
            

        
            # meanPiecaiMetr_auroc= getMeanIgnoreNan(self.picaiLossArr_auroc) # mean(self.picaiLossArr_auroc)
            # meanPiecaiMetr_AP= getMeanIgnoreNan(self.picaiLossArr_AP) # mean(self.picaiLossArr_AP)        
            # meanPiecaiMetr_score= getMeanIgnoreNan(self.picaiLossArr_score) #mean(self.picaiLossArr_score)        





            # print(f"meanPiecaiMetr_auroc {meanPiecaiMetr_auroc} meanPiecaiMetr_AP {meanPiecaiMetr_AP}  meanPiecaiMetr_score {meanPiecaiMetr_score} "  )

            # self.log('val_mean_auroc', meanPiecaiMetr_auroc)
            # self.log('val_mean_AP', meanPiecaiMetr_AP)
            # self.log('val_mean_score', meanPiecaiMetr_score)

            # self.experiment.log_metric('val_mean_auroc', meanPiecaiMetr_auroc)
            # self.experiment.log_metric('val_mean_AP', meanPiecaiMetr_AP)
            # self.experiment.log_metric('val_mean_score', meanPiecaiMetr_score)


            # self.picaiLossArr_auroc_final.append(meanPiecaiMetr_auroc)
            # self.picaiLossArr_AP_final.append(meanPiecaiMetr_AP)
            # self.picaiLossArr_score_final.append(meanPiecaiMetr_score)

            # #resetting to 0 
            # self.picaiLossArr_auroc=[]
            # self.picaiLossArr_AP=[]
            # self.picaiLossArr_score=[]


            #clearing and recreatin temporary directory
            shutil.rmtree(self.temp_val_dir)    
            self.temp_val_dir=tempfile.mkdtemp()
            self.list_gold_val=[]
            self.list_yHat_val=[]


        return {"log": self.log}

    # def validation_step(self, batch, batch_idx):
    #     y_hat, y = self.infer_batch(batch)
    #     loss = self.criterion(y_hat, y)
    #     self.log('val_loss', loss)
    #     return loss

