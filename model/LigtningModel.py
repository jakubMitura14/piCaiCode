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
from monai.metrics import (
    ConfusionMatrixMetric,
    compute_confusion_matrix_metric,
    do_metric_reduction,
    get_confusion_matrix,
)
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
    # gold_im_path = join(directory, patId+ "_gold.npy" )
    # yHat_im_path = join(directory, patId+ "_hat.npy" )
    # np.save(gold_im_path, gold_arr)
    # np.save(yHat_im_path, y_hat_arr)
    gold_im_path = join(directory, patId+ "_gold.nii.gz" )
    yHat_im_path =join(directory, patId+ "_hat.nii.gz" )
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

def save_candidates_to_dir(i,y_true,y_det,patIds,temp_val_dir):
# def save_candidates_to_dir(i,y_true,y_det,patIds,temp_val_dir,reg_hat):
    return saveFilesInDir(y_true[i],y_det[i], temp_val_dir, patIds[i])
    
    # if(reg_hat[i]>0):
    #     return saveFilesInDir(y_true[i],y_det[i], temp_val_dir, patIds[i])
    # #when it is equal 0 we zero out the result
    # return saveFilesInDir(y_true[i],np.zeros_like(y_det[i]), temp_val_dir, patIds[i])    


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
    ,trial
    ):
        super().__init__()
        self.lr = learning_rate
        self.net=net
        # self.modelRegression = UNetToRegresion(2,regression_channels,net)
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.dice_metric = monai.metrics.GeneralizedDiceScore()
        #self.rocAuc=monai.metrics.ROCAUCMetric()
        self.picaiLossArr=[]
        self.post_pred = Compose([ AsDiscrete( to_onehot=2)])
        self.picaiLossArr_auroc=[]
        self.picaiLossArr_AP=[]
        self.picaiLossArr_score=[]
        self.dices=[]
        self.picaiLossArr_auroc_final=picaiLossArr_auroc_final
        self.picaiLossArr_AP_final=picaiLossArr_AP_final
        self.picaiLossArr_score_final=picaiLossArr_score_final
        #temporary directory for validation images and their labels
        self.temp_val_dir= tempfile.mkdtemp() #'/home/sliceruser/data/tempE' #tempfile.mkdtemp()
        self.list_gold_val=[]
        self.list_yHat_val=[]
        self.isAnyNan=False
        #os.makedirs('/home/sliceruser/data/temp')
        # self.postProcess=monai.transforms.Compose([EnsureType(), monai.transforms.ForegroundMask(), AsDiscrete( to_onehot=2)])#, monai.transforms.KeepLargestConnectedComponent()
        # self.postProcess=monai.transforms.Compose([EnsureType(),  monai.transforms.ForegroundMask(), AsDiscrete( to_onehot=2)])#, monai.transforms.KeepLargestConnectedComponent()
        self.postProcess=monai.transforms.Compose([EnsureType(),  monai.transforms.ForegroundMask(), AsDiscrete( to_onehot=2)])#, monai.transforms.KeepLargestConnectedComponent()
        self.postTrue = Compose([EnsureType()])
        #self.F1Score = torchmetrics.F1Score()
        self.lr=lr
        self.trial=trial
        #shutil.rmtree(self.temp_val_dir) 
        #os.makedirs(self.temp_val_dir,  exist_ok = True)             

    def configure_optimizers(self):
        # optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        optimizer = self.optimizer_class(self.parameters())
        return optimizer
    

    
    # def infer_batch_pos(self, batch):
    #     x, y, numLesions = batch["pos"]['chan3_col_name'], batch["pos"]['label'], batch["pos"]['num_lesions_to_retain']
    #     y_hat = self.net(x)
    #     return y_hat, y, numLesions



    # def infer_batch_all(self, batch):
    #     x, numLesions =batch["all"]['chan3_col_name'], batch["all"]['num_lesions_to_retain']
    #     y_hat = self.net(x)
    #     return y_hat, numLesions

    def calcLossHelp(self,isAnythingInAnnotated_list,seg_hat_list, y_true_list,i):
        if(isAnythingInAnnotated_list[i]>0):
            return self.criterion(seg_hat_list[i], y_true_list[i])
        return ' '    
        #     lossReg=F.smooth_l1_loss(torch.Tensor(reg_hat_list[i]).int().to(self.device) , torch.Tensor(int(numLesions_list[i])).int().to(self.device) ) 
        #     return torch.add(lossSeg,lossReg)
        # return  F.smooth_l1_loss(torch.Tensor(reg_hat_list[i]).int().to(self.device) , torch.Tensor(int(numLesions_list[i])).int().to(self.device) ) 



    def calculateLoss(self,isAnythingInAnnotated,seg_hat,y_true,reg_hat,numLesions):
        return self.criterion(seg_hat,y_true)+ F.smooth_l1_loss(reg_hat.flatten(),torch.Tensor(numLesions).to(self.device).flatten() )

        # seg_hat_list = decollate_batch(seg_hat)
        # isAnythingInAnnotated_list = decollate_batch(isAnythingInAnnotated)
        # y_true_list = decollate_batch(y_true)
        # toSum= list(map(lambda i:  self.calcLossHelp(isAnythingInAnnotated_list,seg_hat_list, y_true_list ,i) , list( range(0,len( seg_hat_list)) )))
        # toSum= list(filter(lambda it: it!=' '  ,toSum))
        # if(len(toSum)>0):
        #     segLoss= torch.sum(torch.stack(toSum))
        #     lossReg=F.smooth_l1_loss(reg_hat.flatten(),torch.Tensor(numLesions).to(self.device).flatten())*2
        #     return torch.add(segLoss,lossReg)

        # #print(f"reg_hat {reg_hat} numLesions{numLesions}  "  )        
        # return F.smooth_l1_loss(reg_hat.flatten(),torch.Tensor(numLesions).to(self.device).flatten() )*2


    def training_step(self, batch, batch_idx):
        # every second iteration we will do the training for segmentation
        x, y_true, numLesions,isAnythingInAnnotated = batch['chan3_col_name'], batch['label'], batch['num_lesions_to_retain'], batch['isAnythingInAnnotated']
        # seg_hat, reg_hat = self.modelRegression(x)
        seg_hat = self.net(x)
        return self.criterion(seg_hat,y_true)
        #return self.calculateLoss(isAnythingInAnnotated,seg_hat,y_true,reg_hat,numLesions)

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
        # self.list_gold_val.append(tupl[0])
        # self.list_yHat_val.append(tupl[1])


    def validation_step(self, batch, batch_idx):
        x, y_true, numLesions,isAnythingInAnnotated = batch['chan3_col_name_val'], batch['label_name_val'], batch['num_lesions_to_retain'], batch['isAnythingInAnnotated']
        
        #seg_hat, reg_hat = self.modelRegression(x)        
        # seg_hat, reg_hat = self.modelRegression(x)        
        seg_hat = self.net(x)
        seg_hat=torch.sigmoid(seg_hat)

        #loss= self.criterion(seg_hat,y_true)# self.calculateLoss(isAnythingInAnnotated,seg_hat,y_true,reg_hat,numLesions)      
        #we want only first channel
        print(f"self.postProcess(seg_hat) {self.postProcess(seg_hat)}  y_true {y_true}  ")

        locDice = monai.metrics.compute_generalized_dice( self.postProcess(seg_hat.to(self.device)).to(self.device) ,  y_true.to(self.device)  ).item()
        
        self.dices.append(locDice)
        y_det=y_det[:,1,:,:,:].cpu().detach()
        seg_hat=seg_hat[:,1,:,:,:].cpu().detach()

        y_det = decollate_batch(seg_hat)
        y_true = decollate_batch(y_true)
        patIds = decollate_batch(batch['patient_id'])

        # dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        

        # for i in range(0,len(y_det)):
        #     print(" post process dice")
        #     hatPost=self.postProcess(y_det[i])
        #     # print( f" hatPost {hatPost.size()}  y_true {y_true[i].cpu().size()} " )
        #     print("calc dice")
        #     self.dice_metric(hatPost.cpu() ,y_true[i].cpu())
        #     #monai.metrics.compute_generalized_dice(
        #     # self.rocAuc(hatPost.cpu() ,y_true[i].cpu())
        # print("dice calculated")



        # monai.metrics.compute_confusion_matrix_metric() 
        
        # diceVall = dice_metric.aggregate().item()
        # self.log('loc_dice', diceVall)
        # print("after dices")


        #reg_hat = decollate_batch(reg_hat)
        # print(f" rrrrr prim{reg_hat}  ")

        # reg_hat=np.rint(reg_hat.cpu().detach().numpy().flatten())
        # print(f" rrrrr {reg_hat}  ")
        # print("befor extracting")
        # # y_det=[extract_lesion_candidates( x.cpu().detach().numpy()[1,:,:,:])[0] for x in y_det]
        # y_det=[x.cpu().detach().numpy()[1,:,:,:] for x in y_det]
        # y_true=[x.cpu().detach().numpy()[1,:,:,:] for x in y_true]
        # print("after extracting")
        
        pathssList=[]
        with mp.Pool(processes = mp.cpu_count()) as pool:
            # pathssList=pool.map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir,reg_hat=reg_hat),list(range(0,len(y_true))))
            pathssList=pool.map(partial(save_candidates_to_dir,y_true=y_true,y_det=y_det,patIds=patIds,temp_val_dir=self.temp_val_dir),list(range(0,len(y_true))))
        forGoldVal=list(map(lambda tupl :tupl[0] ,pathssList  ))
        fory_hatVal=list(map(lambda tupl :tupl[1] ,pathssList  ))

#         # self.list_gold_val=self.list_gold_val+forGoldVal
#         # self.list_yHat_val=self.list_gold_val+fory_hatVal

# # save_candidates_to_dir(y_true,y_det,patIds,i,temp_val_dir)
        for i in range(0,len(y_true)):
            # tupl=saveFilesInDir(y_true[i],y_det[i], self.temp_val_dir, patIds[i])
            # print("saving entry   ")
            # self.list_gold_val.append(tupl[0])
            # self.list_yHat_val.append(tupl[1])
            self.list_gold_val.append(forGoldVal[i])
            self.list_yHat_val.append(fory_hatVal[i])

#         self.log('val_loss', loss )

       # return {'loss' :loss,'loc_dice': diceVall }

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

    def validation_epoch_end(self, outputs):
        print("validation_epoch_end")

        self.log('dice', np.mean(self.dices))
        # self.dice_metric.reset()

 
        # print( f"rocAuc  {self.rocAuc.aggregate().item()}"  )
        # #self.log('precision ', monai.metrics.compute_confusion_matrix_metric("precision", confusion_matrix) )
        # self.rocAuc.reset()        


        
        #print(f" self.list_yHat_val {self.list_yHat_val} ")
        if(len(self.list_yHat_val)>1 and (not self.isAnyNan)):
        # if(False):
            valid_metrics = evaluate(y_det=self.list_yHat_val,
                                y_true=self.list_gold_val,
                                num_parallel_calls= os.cpu_count()
                                ,verbose=1
                                #y_true=iter(y_true),
                                ,y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
                                )



            meanPiecaiMetr_auroc=valid_metrics.auroc
            meanPiecaiMetr_AP=valid_metrics.AP
            meanPiecaiMetr_score=(-1)*valid_metrics.score
      
            print(f"meanPiecaiMetr_auroc {meanPiecaiMetr_auroc} meanPiecaiMetr_AP {meanPiecaiMetr_AP}  meanPiecaiMetr_score {meanPiecaiMetr_score} "  )

            self.log('val_mean_auroc', meanPiecaiMetr_auroc)
            self.log('val_mean_AP', meanPiecaiMetr_AP)
            self.log('mean_val_acc', meanPiecaiMetr_score)
            # tensorss = [torch.as_tensor(x['loc_dice']) for x in outputs]
            # if( len(tensorss)>0):
            #     avg_dice = torch.mean(torch.stack(tensorss))

            #     self.log('dice', avg_dice )

            # self.experiment.log_metric('val_mean_auroc', meanPiecaiMetr_auroc)
            # self.experiment.log_metric('val_mean_AP', meanPiecaiMetr_AP)
            # self.experiment.log_metric('val_mean_score', meanPiecaiMetr_score)


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
        # self.temp_val_dir=pathOs.join('/home/sliceruser/data/tempE',str(self.trainer.current_epoch))
        # os.makedirs(self.temp_val_dir,  exist_ok = True)  


        self.list_gold_val=[]
        self.list_yHat_val=[]

        #in case we have Nan values training is unstable and we want to terminate it     
        # if(self.isAnyNan):
        #     self.log('val_mean_score', -0.2)
        #     self.picaiLossArr_score_final=[-0.2]
        #     self.picaiLossArr_AP_final=[-0.2]
        #     self.picaiLossArr_auroc_final=[-0.2]
        #     print(" naans in outputt  ")

        #self.isAnyNan=False
        #return {"mean_val_acc": self.log}


        # # avg_loss = torch.mean(torch.stack([torch.as_tensor(x['val_loss']) for x in outputs]))
        # # print(f"mean_val_loss { avg_loss}")
        # # avg_acc = torch.mean(torch.stack([torch.as_tensor(x['val_acc']) for x in outputs]))
        # #val_accs=list(map(lambda x : x['val_acc'],outputs))
        # val_accs=list(map(lambda x : x['val_acc'].cpu().detach().numpy(),outputs))
        # #print(f" a  val_accs {val_accs} ")
        # val_accs=np.nanmean(np.array( val_accs).flatten())
        # #print(f" b  val_accs {val_accs} mean {np.mean(val_accs)}")

        # #avg_acc = np.mean(np.array(([x['val_acc'].cpu().detach().numpy() for x in outputs])).flatten() )

        # # self.log("mean_val_loss", avg_loss)
        # self.log("mean_val_acc", np.mean(val_accs))

        # # self.log('ptl/val_loss', avg_loss)
        # # self.log('ptl/val_accuracy', avg_acc)
        # #return {'mean_val_loss': avg_loss, 'mean_val_acc':avg_acc}



