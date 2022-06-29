import torch
import pandas as pd
import numpy as np
import torchio as tio
from torch.utils.data import DataLoader
import os
import SimpleITK as sitk
from zipfile import ZipFile
from zipfile import BadZipFile
import dask.dataframe as dd
import os
import multiprocessing as mp
import functools
from functools import partial
import Standardize
import Resampling
import utilsPreProcessing
from utilsPreProcessing import write_to_modif_path 
from registration.elastixRegister import reg_adc_hbv_to_t2w

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('/home/sliceruser/data/metadata/processedMetaData_current.csv')

def resample_adc_hbv_to_t2w(row,secondCol ):
    pathT2w= row['t2w']
    pathh= str(row[secondCol]) 
    newPath = pathh.replace(".mha","_resmaplA.mha" )
    #we check weather resampling was already done if not we do the resampling
    if(len(row[secondCol+'_resmaplA'] )<3):
        imageT2W = sitk.ReadImage(pathT2w)
        targetSpacing = imageT2W.GetSpacing()
        try:
            resampled = Resampling.resample_with_GAN(pathh,targetSpacing)
        except:
            print("error resampling")
        resampled = Resampling.resample_with_GAN(pathh,targetSpacing)

        write_to_modif_path(resampled,pathh,".mha","_resmaplA.mha" )
    return newPath

#needs to be on single thread as resampling GAN is acting on GPU
# we save the metadata to main pandas data frame 
df["adc_resmaplA"]=df.apply(lambda row : resample_adc_hbv_to_t2w(row, 'adc')   , axis = 1) 
df["hbv_resmaplA"]=df.apply(lambda row : resample_adc_hbv_to_t2w(row, 'hbv')   , axis = 1) 
df.to_csv('/home/sliceruser/data/metadata/processedMetaData_current.csv') 
        