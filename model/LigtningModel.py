### Define Data Handling

from distutils.log import error
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
import time

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
    Resize,
    Resized,
    RandSpatialCropd,
        AsDiscrete,
    AsDiscreted,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    SelectItemsd,
    Invertd,
    DivisiblePadd,
    SpatialPadd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandRicianNoised,
    RandFlipd,
    RandAffined,
    ConcatItemsd,
    RandCoarseDropoutd,
    AsDiscreted,
    MapTransform,
    ResizeWithPadOrCropd,
    EnsureChannelFirstd,
    SaveImage,
    EnsureChannelFirst
    
)

import time
from functools import partial
from torchmetrics.functional import precision_recall
from torch.utils.cpp_extension import load
import torchmetrics
# lltm_cuda = load('lltm_cuda', ['lltm_cuda.cpp', 'lltm_cuda_kernel.cu'], verbose=True)
from monai.metrics import (
    ConfusionMatrixMetric,
    compute_confusion_matrix_metric,
    do_metric_reduction,
    get_confusion_matrix,
)

import concurrent.futures
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (Callable, Dict, Hashable, Iterable, List, Optional, Sized,
                    Tuple, Union)

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

try:
    import numpy.typing as npt
except ImportError:  # pragma: no cover
    pass

from picai_eval.analysis_utils import (calculate_dsc, calculate_iou,
                                       label_structure, parse_detection_map)
from picai_eval.image_utils import (read_label, read_prediction,
                                    resize_image_with_crop_or_pad)
from picai_eval.metrics import Metrics

from picai_eval.eval import evaluate_case

# import modelUtlils
import matplotlib.pyplot as plt

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

def getNext(i,results,TIMEOUT):
    try:
        # return it.next(timeout=TIMEOUT)
        return results[i].get(TIMEOUT)

    except Exception as e:
        print(f"timed outt {e} ")
        return None    


def monaiSaveFile(directory,name,arr):
    #Compose(EnsureChannelFirst(),SaveImage(output_dir=directory,separate_folder=False,output_postfix =name) )(arr)
    SaveImage(output_dir=directory,separate_folder=False,output_postfix =name,writer="ITKWriter")(arr)


def getArrayFromPath(path):
    image1=sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image1)


def save_heatmap(arr,dir,name,numLesions,cmapp='gray'):
    path = join(dir,name+'.png')
    arr = np.flip(np.transpose(arr),0)
    plt.imshow(arr , interpolation = 'nearest' , cmap= cmapp)
    plt.title( name+'__'+str(numLesions))
    plt.savefig(path)
    return path


def processDecolated(i,gold_arr,y_hat_arr, directory, studyId,imageArr, postProcess,epoch,regr):
    regr_now = regr[i]
    if(regr_now==0):
        return np.zeros_like(y_hat_arr[i][1,:,:,:])        
    curr_studyId=studyId[i]
    print(f"extracting {curr_studyId}")
    extracted=np.array(extract_lesion_candidates(y_hat_arr[i][1,:,:,:].cpu().detach().numpy())[0]) #, threshold='dynamic'
    print(f"extracted {curr_studyId}")
    return extracted

def iterOverAndCheckType(itemm):
    if(type(itemm) is tuple):
        return list(map(lambda en: en.cpu().detach().numpy(),itemm )) 
    if(torch.is_tensor(itemm)):
        return itemm.cpu().detach().numpy()
    return itemm 

def log_images(i,experiment,golds,extracteds ,t2ws, directory,patIds,epoch,numLesions):
    goldChannel=1
    gold_arr_loc=golds[i]
    maxSlice = max(list(range(0,gold_arr_loc.size(dim=3))),key=lambda ind : torch.sum(gold_arr_loc[goldChannel,:,:,ind]).item() )
    t2w = t2ws[i][0,:,:,maxSlice].cpu().detach().numpy()
    t2wMax= np.max(t2w.flatten())


    curr_studyId=patIds[i]
    gold=golds[i][goldChannel,:,:,maxSlice].cpu().detach().numpy()
    extracted=extracteds[i]

    experiment.log_image( save_heatmap(np.add(t2w.astype('float'),(gold*(t2wMax)).astype('float')),directory,f"gold_plus_t2w_{curr_studyId}_{epoch}",numLesions[i]))
    experiment.log_image( save_heatmap(np.add(gold,((extracted[:,:,maxSlice]>0).astype('int8'))*2),directory,f"gold_plus_extracted_{curr_studyId}_{epoch}",numLesions[i],'plasma'))
    # experiment.log_image( save_heatmap(gold,directory,f"gold_{curr_studyId}_{epoch}",numLesions[i]))


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
    ,trial
    ,dice_final
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.net=net
        self.modelRegression = UNetToRegresion(2,regression_channels,net)
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.dice_metric = monai.metrics.GeneralizedDiceScore()
        self.picaiLossArr=[]
        self.post_pred = Compose([ AsDiscrete( to_onehot=2)])
        self.picaiLossArr_auroc=[]
        self.picaiLossArr_AP=[]
        self.picaiLossArr_score=[]
        self.dices=[]
        self.surfDists=[]
        self.dice_final=dice_final        
        self.picaiLossArr_auroc_final=picaiLossArr_auroc_final
        self.picaiLossArr_AP_final=picaiLossArr_AP_final
        self.picaiLossArr_score_final=picaiLossArr_score_final
        self.temp_val_dir= '/home/sliceruser/locTemp/tempH' #tempfile.mkdtemp()
        self.list_gold_val=[]
        self.list_yHat_val=[]
        self.ldiceLocst_back_yHat_val=[]
        self.isAnyNan=False
        #os.makedirs('/home/sliceruser/data/temp')
        # self.postProcess=monai.transforms.Compose([EnsureType(), monai.transforms.ForegroundMask(), AsDiscrete( to_onehot=2)])#, monai.transforms.KeepLargestConnectedComponent()
        # self.postProcess=monai.transforms.Compose([EnsureType(),  monai.transforms.ForegroundMask(), AsDiscrete( to_onehot=2)])#, monai.transforms.KeepLargestConnectedComponent()
        self.postProcess=monai.transforms.Compose([EnsureType(),EnsureChannelFirst(), AsDiscrete(argmax=True, to_onehot=2)])#, monai.transforms.KeepLargestConnectedComponent()
        self.postTrue = Compose([EnsureType()])
        self.regLoss = nn.BCEWithLogitsLoss()
        #self.F1Score = torchmetrics.F1Score()

        os.makedirs(self.temp_val_dir,  exist_ok = True)             
        shutil.rmtree(self.temp_val_dir) 
        os.makedirs(self.temp_val_dir,  exist_ok = True)             

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.learning_rate)
        # hyperparameters from https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10, T_mult=1, eta_min=0.001, last_epoch=-1 )
        return [optimizer], [lr_scheduler]

    def infer_train_ds_labels(self, batch):
        x, y, numLesions = batch['chan3_col_name'] , batch['label'], batch['num_lesions_to_retain']
        segmMap,regr = self.modelRegression(x)
        return segmMap,regr, y, numLesions
    # def infer_train_ds_labels(self, batch):
    #     x, y, numLesions = batch["train_ds_labels"]['chan3_col_name'] , batch["train_ds_labels"]['label'], batch["train_ds_labels"]['num_lesions_to_retain']
    #     segmMap,regr = self.modelRegression(x)
    #     return segmMap,regr, y, numLesions


    # def infer_train_ds_no_labels(self, batch):
    #     x, numLesions =batch["train_ds_no_labels"]['chan3_col_name'],batch["train_ds_no_labels"]['num_lesions_to_retain']
    #     segmMap,regr = self.modelRegression(x)
    #     return regr, numLesions


    def training_step(self, batch, batch_idx):
        # every second iteration we will do the training for segmentation

        seg_hat,reg_hat, y_true, numLesions=self.infer_train_ds_labels( batch)
        # regr_no_lab, numLesions_no_lab= self.infer_train_ds_no_labels( batch) 

        return torch.sum(torch.stack([self.criterion(seg_hat,y_true)
                                    ,self.regLoss(reg_hat.flatten().float(),torch.Tensor(numLesions).to(self.device).flatten().float() ) 
                                    # ,self.regLoss(regr_no_lab.flatten(),torch.Tensor(numLesions_no_lab).to(self.device).flatten() ) 
                                        ]))


    def validation_step(self, batch, batch_idx):
        x, y_true_prim, numLesions,isAnythingInAnnotated = batch['chan3_col_name_val'], batch['label_name_val'], batch['num_lesions_to_retain'], batch['isAnythingInAnnotated']
        numBatches = y_true_prim.size(dim=0)
        #seg_hat, reg_hat = self.modelRegression(x)        
        # seg_hat, reg_hat = self.modelRegression(x)        
        seg_hat,regr = self.modelRegression(x)
        seg_hat = seg_hat.cpu().detach()
        regr=regr.cpu().detach().numpy()
        # regr= list(map(lambda el : int(el>0.5) ,regr ))
        seg_hat=torch.sigmoid(seg_hat).cpu().detach()
        t2wb=decollate_batch(batch['t2wb'])
        labelB=decollate_batch(batch['labelB'])
        #loss= self.criterion(seg_hat,y_true)# self.calculateLoss(isAnythingInAnnotated,seg_hat,y_true,reg_hat,numLesions)      
        y_det = decollate_batch(seg_hat.cpu().detach())
        # y_background = decollate_batch(seg_hat[:,0,:,:,:].cpu().detach())
        y_true = decollate_batch(y_true_prim.cpu().detach())
        patIds = decollate_batch(batch['study_id'])
        numLesions = decollate_batch(batch['num_lesions_to_retain'])
        images = decollate_batch(x.cpu().detach()) 

        # print(f"val num batches {numBatches} t2wb {t2wb} patIds {patIds} labelB {labelB}")
        print(f"val num batches {numBatches} ")
        lenn=numBatches
        processedCases=[]
        my_task=partial(processDecolated,gold_arr=y_true,y_hat_arr=y_det,directory= self.temp_val_dir,studyId= patIds
                    ,imageArr=images, postProcess=self.postProcess,epoch=self.current_epoch,regr=regr)
        with mp.Pool(processes = mp.cpu_count()) as pool:
            #it = pool.imap(my_task, range(lenn))
            results = list(map(lambda i: pool.apply_async(my_task, (i,)) ,list(range(lenn))  ))
            time.sleep(45)
            processedCases=list(map(lambda ind :getNext(ind,results,10) ,list(range(lenn)) ))

        isTaken= list(map(lambda it:type(it) != type(None),processedCases))
        extracteds=list(filter(lambda it:type(it) != type(None),processedCases))

        lenn=len(extracteds)
        print(f"lenn after extract {lenn}")
        # extracteds=list(filter(lambda it:it.numpy(),extracteds))


        # processedCases=list(map(partial(processDecolated,gold_arr=y_true,y_hat_arr=y_det,directory= self.temp_val_dir,studyId= patIds
        #             ,imageArr=images, experiment=self.logger.experiment,postProcess=self.postProcess,epoch=self.current_epoch)
        #             ,range(0,numBatches)))
        
        if(len(extracteds)>0):
            print("inside if   ")
            directory= self.temp_val_dir
            print("inside if  b ")
            experiment=self.logger.experiment
            print("inside if  c ")            
            epoch=self.current_epoch
            print("start logging")
            list(map(partial(log_images
                ,experiment=experiment,golds=y_true,extracteds=extracteds 
                ,t2ws=images,directory=directory ,patIds=patIds,epoch=epoch,numLesions=numLesions),range(lenn)))
            # y_true= list(map(lambda el: el.numpy()  ,y_true))                                              
            print("end logging")
            valid_metrics = evaluate(y_det=extracteds,
                                    y_true=list(map(lambda el: el.numpy()[1,:,:,:]  ,y_true)),
                                    num_parallel_calls= os.cpu_count()
                                    ,verbose=1)
            meanPiecaiMetr_auroc=0.0 if math.isnan(valid_metrics.auroc) else valid_metrics.auroc
            meanPiecaiMetr_AP=0.0 if math.isnan(valid_metrics.AP) else valid_metrics.AP
            meanPiecaiMetr_score= 0.0 if math.isnan(valid_metrics.score) else  valid_metrics.score

            print("start dice")
            extracteds= list(map(lambda numpyEntry : self.postProcess(torch.from_numpy((numpyEntry>0).astype('int8'))) ,extracteds  ))
            extracteds= torch.stack(extracteds)

            # extracteds= self.postProcess(extracteds)#argmax=True,
            y_true=list(filter(lambda tupl:  isTaken[tupl[0]] , enumerate(y_true)))
            y_true=list(map(lambda tupl:  tupl[1] ,y_true))

            golds=torch.stack(y_true).cpu()
            # print(f"get dice  extrrr {extracteds.cpu()}  Y true  {y_true_prim.cpu()}   ")
            diceLoc=0.0
            try:
                diceLoc=monai.metrics.compute_generalized_dice( extracteds.cpu() ,golds)[1].item()
            except:
                pass    
            print(f"diceLoc {diceLoc}")

            # gold = list(map(lambda tupl: tupl[2] ,processedCases ))

            return {'dices': diceLoc, 'meanPiecaiMetr_auroc':meanPiecaiMetr_auroc
                    ,'meanPiecaiMetr_AP' :meanPiecaiMetr_AP,'meanPiecaiMetr_score': meanPiecaiMetr_score}

        return {'dices': 0.0, 'meanPiecaiMetr_auroc':0.0
                ,'meanPiecaiMetr_AP' :0.0,'meanPiecaiMetr_score': 0.0}




    def validation_epoch_end(self, outputs):
        print("validation_epoch_end")

        allDices = np.array(([x['dices'] for x in outputs])).flatten() 
        allmeanPiecaiMetr_auroc = np.array(([x['meanPiecaiMetr_auroc'] for x in outputs])).flatten() 
        allmeanPiecaiMetr_AP = np.array(([x['meanPiecaiMetr_AP'] for x in outputs])).flatten() 
        allmeanPiecaiMetr_score = np.array(([x['meanPiecaiMetr_score'] for x in outputs])).flatten() 
        
    
        # allDices = np.array(([x['dices'].cpu().detach().numpy() for x in outputs])).flatten() 
        # allmeanPiecaiMetr_auroc = np.array(([x['meanPiecaiMetr_auroc'].cpu().detach().numpy() for x in outputs])).flatten() 
        # allmeanPiecaiMetr_AP = np.array(([x['meanPiecaiMetr_AP'].cpu().detach().numpy() for x in outputs])).flatten() 
        # allmeanPiecaiMetr_score = np.array(([x['meanPiecaiMetr_score'].cpu().detach().numpy() for x in outputs])).flatten() 
        
    
        
        
        if(len(allDices)>0):            
            meanPiecaiMetr_auroc=np.nanmean(allmeanPiecaiMetr_auroc)
            meanPiecaiMetr_AP=np.nanmean(allmeanPiecaiMetr_AP)
            meanPiecaiMetr_score= np.nanmean(allmeanPiecaiMetr_score)

            self.log('dice', (-1)*np.nanmean(allDices))

            print(f"meanPiecaiMetr_auroc {meanPiecaiMetr_auroc} meanPiecaiMetr_AP {meanPiecaiMetr_AP}  meanPiecaiMetr_score {meanPiecaiMetr_score} "  )

            self.log('val_mean_auroc', meanPiecaiMetr_auroc)
            self.log('val_mean_AP', meanPiecaiMetr_AP)
            self.log('mean_val_acc', meanPiecaiMetr_score)

            self.picaiLossArr_auroc_final.append(meanPiecaiMetr_auroc)
            self.picaiLossArr_AP_final.append(meanPiecaiMetr_AP)
            self.picaiLossArr_score_final.append(meanPiecaiMetr_score)
            self.dice_final.append((-1)*np.nanmean(allDices))

 















# def evaluate_all_cases(listPerEval):
#     case_target: Dict[Hashable, int] = {}
#     case_weight: Dict[Hashable, float] = {}
#     case_pred: Dict[Hashable, float] = {}
#     lesion_results: Dict[Hashable, List[Tuple[int, float, float]]] = {}
#     lesion_weight: Dict[Hashable, List[float]] = {}

#     meanPiecaiMetr_auroc=0.0
#     meanPiecaiMetr_AP=0.0
#     meanPiecaiMetr_score=0.0

#     idx=0
#     if(len(listPerEval)>0):
#         for pairr in listPerEval:
#             idx+=1
#             lesion_results_case, case_confidence = pairr

#             case_weight[idx] = 1.0
#             case_pred[idx] = case_confidence
#             if len(lesion_results_case):
#                 case_target[idx] = np.max([a[0] for a in lesion_results_case])
#             else:
#                 case_target[idx] = 0

#             # accumulate outputs
#             lesion_results[idx] = lesion_results_case
#             lesion_weight[idx] = [1.0] * len(lesion_results_case)

#         # collect results in a Metrics object
#         valid_metrics = Metrics(
#             lesion_results=lesion_results,
#             case_target=case_target,
#             case_pred=case_pred,
#             case_weight=case_weight,
#             lesion_weight=lesion_weight
#         )
#         # for i in range(0,numIters):
#         #     valid_metrics = evaluate(y_det=self.list_yHat_val[i*numPerIter:min((i+1)*numPerIter,lenn)],
#         #                         y_true=self.list_gold_val[i*numPerIter:min((i+1)*numPerIter,lenn)],
#         #                         num_parallel_calls= min(numPerIter,os.cpu_count())
#         #                         ,verbose=1
#         #                         #,y_true_postprocess_func=lambda pred: pred[1,:,:,:]
#         #                         #y_true=iter(y_true),
#         #                         ,y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
#         #                         #,y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
#         #                         )
#         # meanPiecaiMetr_auroc_list.append(valid_metrics.auroc)
#         # meanPiecaiMetr_AP_list.append(valid_metrics.AP)
#         # meanPiecaiMetr_score_list.append((-1)*valid_metrics.score)
#         #print("finished evaluating")

#         meanPiecaiMetr_auroc=valid_metrics.auroc
#         meanPiecaiMetr_AP=valid_metrics.AP
#         meanPiecaiMetr_score=(-1)*valid_metrics.score
#     return (meanPiecaiMetr_auroc,meanPiecaiMetr_AP,meanPiecaiMetr_score )    





# def saveFilesInDir(gold_arr,y_hat_arr, directory, patId,imageArr, hatPostA):
#     """
#     saves arrays in given directory and return paths to them
#     """
#     adding='_e'
#     monaiSaveFile(directory,patId+ "_gold"+adding,gold_arr)
#     monaiSaveFile(directory,patId+ "_hat"+adding,y_hat_arr)
#     monaiSaveFile(directory,patId+ "image"+adding,imageArr)
#     monaiSaveFile(directory,patId+ "imageB"+adding,imageArr)
#     monaiSaveFile(directory,patId+ "hatPostA"+adding,hatPostA)

#     # gold_im_path = join(directory, patId+ "_gold.npy" )
#     # yHat_im_path = join(directory, patId+ "_hat.npy" )
#     # np.save(gold_im_path, gold_arr)
#     # np.save(yHat_im_path, y_hat_arr)
#     gold_im_path = join(directory, patId+ "_gold.nii.gz" )
#     yHat_im_path =join(directory, patId+ "_hat.nii.gz" )
#     image_path =join(directory, patId+ "image.nii.gz" )
#     imageB_path =join(directory, patId+ "imageB.nii.gz" )
#     hatPostA_path =join(directory, patId+ "hatPostA.nii.gz" )
#     # print(f"suuum image {torch.sum(imageArr)}    suum hat  {np.sum( y_hat_arr.numpy())} hatPostA {np.sum(hatPostA)} hatPostA uniqq {np.unique(hatPostA) } hatpostA shape {hatPostA.shape} y_hat_arr sh {y_hat_arr.shape} gold_arr shape {gold_arr.shape} ")
#     print(f" suum hat  {np.sum( y_hat_arr.numpy())} gold_arr chan 0 sum  {np.sum(gold_arr[0,:,:,:].numpy())} chan 1 sum {np.sum(gold_arr[1,:,:,:].numpy())} hatPostA chan 0 sum  {np.sum(hatPostA[0,:,:,:])} chan 1 sum {np.sum(hatPostA[1,:,:,:])}    ")
#     # gold_arr=np.swapaxes(gold_arr,0,2)
#     # y_hat_arr=np.swapaxes(y_hat_arr,0,2)
#     # print(f"uniq gold { gold_arr.shape  }   yhat { y_hat_arr.shape }   yhat maxes  {np.maximum(y_hat_arr)}  hyat min {np.minimum(y_hat_arr)} ")
#     gold_arr=gold_arr[1,:,:,:].numpy()
#     # gold_arr=np.flip(gold_arr,(1,0))
#     y_hat_arr=y_hat_arr[1,:,:,:].numpy()

#     gold_arr=np.swapaxes(gold_arr,0,2)
#     y_hat_arr=np.swapaxes(y_hat_arr,0,2)
    
#     image = sitk.GetImageFromArray(gold_arr)
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(gold_im_path)
#     writer.Execute(image)


#     image = sitk.GetImageFromArray(y_hat_arr)
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(yHat_im_path)
#     writer.Execute(image) 

#     image = sitk.GetImageFromArray(  np.swapaxes(imageArr[0,:,:,:].numpy(),0,2) ) 
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(image_path)
#     writer.Execute(image)

#     image = sitk.GetImageFromArray(  np.swapaxes(imageArr[1,:,:,:].numpy(),0,2) )
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(imageB_path)
#     writer.Execute(image)

#     image = sitk.GetImageFromArray(np.swapaxes(hatPostA[1,:,:,:],0,2))
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(hatPostA_path)
#     writer.Execute(image)




#     return(gold_im_path,yHat_im_path)






    #     # return {'dices': dices, 'extrCases0':extrCases0,'extrCases1':extrCases1, 'extrCases2':extrCases2 }
    # def processOutputs(self,outputs):
    #     listt = [x['from_case'] for x in outputs] 
    #     listt =[item for sublist in listt for item in sublist]
    #     print(f"listt b {listt}" )
    #     listt= list(map(iterOverAndCheckType, listt))
    #     print(f"listt c {listt}" )
    #     return listt









#         # print( f"rocAuc  {self.rocAuc.aggregate().item()}"  )
#         # #self.log('precision ', monai.metrics.compute_confusion_matrix_metric("precision", confusion_matrix) )
#         # self.rocAuc.reset()        


        
#         print(f" num to validate  { len(self.list_yHat_val)} ")
#         if(len(self.list_yHat_val)>0 ): #and (not self.isAnyNan)
#         # if(False):
#             # with mp.Pool(processes = mp.cpu_count()) as pool:
#             #     dices=pool.map(partial(calcDiceFromPaths,list_yHat_val=self.list_yHat_val,list_gold_val=self.list_gold_val   ),list(range(0,len(self.list_yHat_val))))
#             # dices=list(map(partial(calcDiceFromPaths,list_yHat_val=self.list_yHat_val,list_gold_val=self.list_gold_val   ),list(range(0,len(self.list_yHat_val)))))
#             #meanDice=torch.mean(torch.stack( dices)).item()
#             meanDice=np.mean( self.dices)
#             self.log('meanDice',np.mean( self.dices))
#             print(f"meanDice {meanDice} ")
#             # self.log('meanDice',torch.mean(torch.stack( self.dices)).item() )
#             # print('meanDice',np.mean( np.array(self.dices ).flatten()))
#             # self.log('mean_surface_distance',torch.mean(torch.stack( self.surfDists)).item())

#             lenn=len(self.list_yHat_val)
#             numPerIter=1
#             numIters=math.ceil(lenn/numPerIter)-1



#             meanPiecaiMetr_auroc_list=[]
#             meanPiecaiMetr_AP_list=[]
#             meanPiecaiMetr_score_list=[]
#             print(f" numIters {numIters} ")
            
#             pool = mp.Pool()
#             listPerEval=[None] * lenn

#             # #timeout based on https://stackoverflow.com/questions/66051638/set-a-time-limit-on-the-pool-map-operation-when-using-multiprocessing
#             my_task=partial(evaluate_case_for_map,y_det= self.list_yHat_val,y_true=self.list_gold_val)
#             # def my_callback(t):
#             #     print(f"tttttt  {t}")
#             #     s, i = t
#             #     listPerEval[i] = s
#             # results=[pool.apply_async(my_task, args=(i,), callback=my_callback) for i in list(range(0,lenn))]
#             # TIMEOUT = 300# second timeout
#             # time.sleep(TIMEOUT)
#             # pool.terminate()
#             # #filtering out those that timed out
#             # listPerEval=list(filter(lambda it:it!=None,listPerEval))
#             # print(f" results timed out {lenn-len(listPerEval)} from all {lenn} ")

#             TIMEOUT = 50# second timeout


# # TIMEOUT = 2# second timeout
# # with mp.Pool(processes = mp.cpu_count()) as pool:
# #     results = list(map(lambda i: pool.apply_async(my_task, (i,)) ,list(range(lenn))  ))
    
# #     for i in range(lenn):
# #         try:
# #             return_value = results[i].get(2) # wait for up to time_to_wait seconds
# #         except mp.TimeoutError:
# #             print('Timeout for v = ', i)
# #         else:
# #             squares[i]=return_value
# #             print(f'Return value for v = {i} is {return_value}')


# #     # it = pool.imap(my_task, range(lenn))
# #     # squares=list(map(lambda ind :getNext(it,TIMEOUT) ,list(range(lenn)) ))
# # print(squares)


#             with mp.Pool(processes = mp.cpu_count()) as pool:
#                 #it = pool.imap(my_task, range(lenn))
#                 results = list(map(lambda i: pool.apply_async(my_task, (i,)) ,list(range(lenn))  ))
#                 time.sleep(TIMEOUT)
#                 listPerEval=list(map(lambda ind :getNext(ind,results,5) ,list(range(lenn)) ))
#             #filtering out those that timed out
#             listPerEval=list(filter(lambda it:it!=None,listPerEval))
#             print(f" results timed out {lenn-len(listPerEval)} from all {lenn} ")                
#                     # pathssList=pool.map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir,reg_hat=reg_hat),list(range(0,len(y_true))))
#                 # listPerEval=pool.map( partial(evaluate_case_for_map,y_det= self.list_yHat_val,y_true=self.list_gold_val) , list(range(0,lenn)))


#             # listPerEval=list(map( partial(evaluate_case_for_map,y_det= self.list_yHat_val,y_true=self.list_gold_val) , list(range(0,lenn))))


#             # initialize placeholders

#             # meanPiecaiMetr_auroc=np.nanmean(meanPiecaiMetr_auroc_list)
#             # meanPiecaiMetr_AP=np.nanmean(meanPiecaiMetr_AP_list)
#             # meanPiecaiMetr_score=np.nanmean(meanPiecaiMetr_score_list)
        

      
#             print(f"meanPiecaiMetr_auroc {meanPiecaiMetr_auroc} meanPiecaiMetr_AP {meanPiecaiMetr_AP}  meanPiecaiMetr_score {meanPiecaiMetr_score} "  )

#             self.log('val_mean_auroc', meanPiecaiMetr_auroc)
#             self.log('val_mean_AP', meanPiecaiMetr_AP)
#             self.log('mean_val_acc', meanPiecaiMetr_score)
#             # tensorss = [torch.as_tensor(x['loc_dice']) for x in outputs]
#             # if( len(tensorss)>0):
#             #     avg_dice = torch.mean(torch.stack(tensorss))

#             self.picaiLossArr_auroc_final.append(meanPiecaiMetr_auroc)
#             self.picaiLossArr_AP_final.append(meanPiecaiMetr_AP)
#             self.picaiLossArr_score_final.append(meanPiecaiMetr_score)

#             #resetting to 0 
#             self.picaiLossArr_auroc=[]
#             self.picaiLossArr_AP=[]
#             self.picaiLossArr_score=[]







#         #clearing and recreatin temporary directory
#         #shutil.rmtree(self.temp_val_dir)   
#         #self.temp_val_dir=tempfile.mkdtemp() 
#         self.temp_val_dir=pathOs.join('/home/sliceruser/data/tempH',str(self.trainer.current_epoch))
#         os.makedirs(self.temp_val_dir,  exist_ok = True)  


#         self.list_gold_val=[]
#         self.list_yHat_val=[]
#         self.list_back_yHat_val=[]

#         #in case we have Nan values training is unstable and we want to terminate it     
#         # if(self.isAnyNan):
#         #     self.log('val_mean_score', -0.2)
#         #     self.picaiLossArr_score_final=[-0.2]
#         #     self.picaiLossArr_AP_final=[-0.2]
#         #     self.picaiLossArr_auroc_final=[-0.2]
#         #     print(" naans in outputt  ")

#         #self.isAnyNan=False
#         #return {"mean_val_acc": self.log}


#         # # avg_loss = torch.mean(torch.stack([torch.as_tensor(x['val_loss']) for x in outputs]))
#         # # print(f"mean_val_loss { avg_loss}")
#         # # avg_acc = torch.mean(torch.stack([torch.as_tensor(x['val_acc']) for x in outputs]))
#         # #val_accs=list(map(lambda x : x['val_acc'],outputs))
#         # val_accs=list(map(lambda x : x['val_acc'].cpu().detach().numpy(),outputs))
#         # #print(f" a  val_accs {val_accs} ")
#         # val_accs=np.nanmean(np.array( val_accs).flatten())
#         # #print(f" b  val_accs {val_accs} mean {np.mean(val_accs)}")

#         # #avg_acc = np.mean(np.array(([x['val_acc'].cpu().detach().numpy() for x in outputs])).flatten() )

#         # # self.log("mean_val_loss", avg_loss)
#         # self.log("mean_val_acc", np.mean(val_accs))

#         # # self.log('ptl/val_loss', avg_loss)
#         # # self.log('ptl/val_accuracy', avg_acc)
#         # #return {'mean_val_loss': avg_loss, 'mean_val_acc':avg_acc}

# #self.postProcess

# #             image1=sitk.ReadImage(path)
# # #     data = sitk.GetArrayFromImage(image1)



# def save_candidates_to_dir(i,y_true,y_det,patIds,temp_val_dir,images,hatPostA):
# # def save_candidates_to_dir(i,y_true,y_det,patIds,temp_val_dir,reg_hat):
#     return saveFilesInDir(y_true[i],y_det[i], temp_val_dir, patIds[i],images[i],hatPostA[i])
    

# def evaluate_case_for_map(i,y_det,y_true):
#     pred=sitk.GetArrayFromImage(sitk.ReadImage(y_det[i]))
#     pred=extract_lesion_candidates(pred)[0]
#     image = sitk.GetImageFromArray(  np.swapaxes(pred,0,2) )
#     writer = sitk.ImageFileWriter()
#     writer.SetFileName(y_det[i].replace(".nii.gz", "_extracted.nii.gz   "))
#     writer.Execute(image)

#     print("evaluate_case_for_map") 
#     return evaluate_case(y_det=y_det[i] 
#                         ,y_true=y_true[i] 
#                         ,y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])

# def getNext(i,results,TIMEOUT):
#     try:
#         # return it.next(timeout=TIMEOUT)
#         return results[i].get(TIMEOUT)

#     except:
#         print("timed outt ")
#         return None    


# def processDice(i,postProcess,y_det,y_true):
#     hatPost=postProcess(y_det[i])
#     # print( f" hatPost {hatPost.size()}  y_true {y_true[i].cpu().size()} " )
#     locDice=monai.metrics.compute_generalized_dice( hatPost ,y_true[i])[1].item()
#     print(f"locDice {locDice}")
#     return (locDice,hatPost.numpy())




        # pathssList=[]
        # dicesList=[]
        # hatPostA=[]
        # # with mp.Pool(processes = mp.cpu_count()) as pool:
        # #     # pathssList=pool.map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir,reg_hat=reg_hat),list(range(0,len(y_true))))
        # #     dicesList=pool.map(partial(processDice,postProcess=self.postProcess,y_det=y_det, y_true=y_true ),list(range(0,len(y_true))))
        # dicesList=list(map(partial(processDice,postProcess=self.postProcess,y_det=y_det, y_true=y_true ),list(range(0,len(y_true)))))

        # hatPostA=list(map(lambda tupl: tupl[1],dicesList ))
        # dicees=list(map(lambda tupl: tupl[0],dicesList ))
        # # self.logger.experiment.

        # # with mp.Pool(processes = mp.cpu_count()) as pool:        
        # #     pathssList=pool.map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir,images=images,hatPostA=hatPostA),list(range(0,len(y_true))))

        # pathssList=list(map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir,images=images,hatPostA=hatPostA),list(range(0,len(y_true)))))

        # forGoldVal=list(map(lambda tupl :tupl[0] ,pathssList  ))
        # fory_hatVal=list(map(lambda tupl :tupl[1] ,pathssList  ))
        # # fory__bach_hatVal=list(map(lambda tupl :tupl[2] ,pathssList  ))

        # for i in range(0,len(y_true)):
            
        #     # tupl=saveFilesInDir(y_true[i],y_det[i], self.temp_val_dir, patIds[i])
        #     # print("saving entry   ")
        #     # self.list_gold_val.append(tupl[0])
        #     # self.list_yHat_val.append(tupl[1])
        #     self.list_gold_val.append(forGoldVal[i])
        #     self.list_yHat_val.append(fory_hatVal[i])
        #     self.dices.append(dicees[i])
            # self.list_back_yHat_val.append(fory__bach_hatVal[i])
# #         self.log('val_loss', loss )

#        # return {'loss' :loss,'loc_dice': diceVall }

        #TODO probably this [1,:,:,:] could break the evaluation ...
        # y_det=[x.cpu().detach().numpy()[1,:,:,:][0] for x in y_det]
        # y_true=[x.cpu().detach().numpy() for x in y_true]
        # y_det= list(map(self.postProcess  , y_det))
        # y_true= list(map(self.postTrue , y_det))


        # if(torch.sum(torch.isnan( y_det))>0):
        #     self.isAnyNan=True

        # regress_res2= torch.flatten(reg_hat) 
        # regress_res3=list(map(lambda el:round(el) ,torch.flatten(regress_res2).cpu().detach().numpy() ))

        # total_loss=precision_recall(torch.Tensor(regress_res3).int(), torch.Tensor(numLesions).cpu().int(), average='macro', num_classes=4)
        # total_loss1=torch.mean(torch.stack([total_loss[0],total_loss[1]] ))#self.F1Score
        
        # if(torch.sum(isAnythingInAnnotated)>0):
        #     dice = DiceMetric()
        #     for i in range(0,len( y_det)):
        #         if(isAnythingInAnnotated[i]>0):
        #             y_det_i=self.postProcess(y_det[i])[0,:,:,:].cpu()
        #             y_true_i=self.postTrue(y_true[i])[1,:,:,:].cpu()
        #             if(torch.sum(y_det_i).item()>0 and torch.sum(y_true_i).item()>0 ):
        #                 dice(y_det_i,y_true_i)

        #     self.log("dice", dice.aggregate())
        #     #print(f" total loss a {total_loss1} val_loss {val_losss}  dice.aggregate() {dice.aggregate()}")
        #     total_loss2= torch.add(total_loss1,dice.aggregate())
        #     print(f" total loss b {total_loss2}  total_loss,dice.aggregate() {dice.aggregate()}")
            
        #     self.picaiLossArr_score_final.append(total_loss2.item())
        #     return {'val_acc': total_loss2.item(), 'val_loss':val_losss}
        
        # #in case no positive segmentation information is available
        # self.picaiLossArr_score_final.append(total_loss1.item())
        # return {'val_acc': total_loss1.item(), 'val_loss':val_losss}


    #return {'dices': dices, 'extrCases':extrCases}