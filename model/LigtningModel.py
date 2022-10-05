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




# def my_task(v):
#     time.sleep(v)
#     return v ** 2


# lenn=8
# squares=[None] * lenn

# TIMEOUT = 2# second timeout
# with mp.Pool(processes = mp.cpu_count()) as pool:
#     results = list(map(lambda i: pool.apply_async(my_task, (i,)) ,list(range(lenn))  ))
    
#     for i in range(lenn):
#         try:
#             return_value = results[i].get(2) # wait for up to time_to_wait seconds
#         except mp.TimeoutError:
#             print('Timeout for v = ', i)
#         else:
#             squares[i]=return_value
#             print(f'Return value for v = {i} is {return_value}')

#     # it = pool.imap(my_task, range(lenn))
#     # squares=list(map(lambda ind :getNext(it,TIMEOUT) ,list(range(lenn)) ))
# print(squares)



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
        self.criterion = criterion
        self.optimizer_class = optimizer_class
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.experiment=experiment
        self.picaiLossArr=[]
        self.post_pred = Compose([ AsDiscrete( to_onehot=2)])
#        self.post_pred = Compose([ AsDiscrete(argmax=True, to_onehot=2)])
        self.dices=[]

        #self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        #self.post_label = Compose([EnsureType("tensor", device="cpu"), torchio.transforms.OneHot(include=["label"] ,num_classes=2)])
        #self.post_label = Compose([EnsureType("tensor", device="cpu"), torchio.transforms.OneHot(include=["label"] ,num_classes=2)])
        self.picaiLossArr_auroc=[]
        self.picaiLossArr_AP=[]
        self.picaiLossArr_score=[]
        self.postProcess=monai.transforms.Compose([EnsureType(),  monai.transforms.ForegroundMask(), AsDiscrete( to_onehot=2)])#, monai.transforms.KeepLargestConnectedComponent()
        self.postTrue = Compose([EnsureType()])
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
        #marking that we had some Nan numbers in the tensor
        if(torch.sum(torch.isnan( y_det))>0):
            self.isAnyNan=True
        
        loss = self.criterion(y_det, y_true)
        y_det=torch.sigmoid(y_det)
        # print( f"before extract lesion  sum a {torch.sum(y_hat)  } " )

        y_det = decollate_batch(y_det.cpu())
        y_true = decollate_batch(y_true.cpu())
        patIds = decollate_batch(patIds)
        #print(f"after decollate  y_hat{y_hat[0].size()} labels{labels[0].size()} y_hat len {len(y_hat)} labels len {len(labels)}")
        
        for i in range(0,len(y_det)):
            hatPost=self.postProcess(y_det[i])
            # print( f" hatPost {hatPost.size()}  y_true {y_true[i].cpu().size()} " )
            locDice=monai.metrics.compute_generalized_dice( hatPost ,y_true[i])
            #monai.metrics.compute_generalized_dice(
            # self.rocAuc(hatPost.cpu() ,y_true[i].cpu())
            self.dices.append(locDice)            
        
        y_det=[extract_lesion_candidates( x.cpu().detach().numpy()[1,:,:,:])[0] for x in y_det]
        y_true=[x.cpu().detach().numpy()[1,:,:,:] for x in y_true]


        for i in range(0,len(y_true)):
            tupl=saveFilesInDir(y_true[i],y_det[i], self.temp_val_dir, patIds[i])
            self.list_gold_val.append(tupl[0])
            self.list_yHat_val.append(tupl[1])
        #now we need to save files in temporary direcory and save outputs to the appripriate lists wit paths
    

        self.log('val_loss', loss)

        return loss




    def validation_epoch_end(self, outputs):
        """
        just in order to log the dice metric on validation data 
        """

        if(len(self.list_yHat_val)>1 and (not self.isAnyNan)):
            self.log('meanDice',torch.mean(torch.stack( self.dices)).item())
            try:
                
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
            except:
                print("error in evall")    
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

