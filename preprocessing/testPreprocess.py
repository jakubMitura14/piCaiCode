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
import os.path
from os import path as pathOs
import comet_ml
from comet_ml import Experiment


experiment = Experiment(
    api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
    #workspace="picai", # Optional
    project_name="picai_first_preprocessing", # Optional
    #experiment_name="baseline" # Optional
)


df = pd.read_csv('/home/sliceruser/data/metadata/processedMetaData.csv')
#currently We want only imagfes with associated masks
df = df.loc[df['isAnyMissing'] ==False]
df = df.loc[df['isAnythingInAnnotated']>0 ]
# ignore all with deficient spacing
for keyWord in ['t2w','adc', 'cor','hbv','sag'  ]:    
    colName=keyWord+ "_spac_x"
    df = df.loc[df[colName]>0 ]
# get only complete representaions and only those with labels
#df = df.loc[df['isAnyMissing'] ==False]
df = df.loc[df['isAnythingInAnnotated']>0 ]    
print(df)    
#just for testing    
#df=df.sample(n = 4)#TODO remove
#df= df.head(4)
##df.to_csv('/home/sliceruser/data/metadata/processedMetaData_current.csv') 

########## Standarization


trainedModelsBasicPath='/home/sliceruser/data/preprocess/standarizationModels'
for keyWord in ['t2w','adc', 'cor','hbv','sag'  ]:
    df[keyWord+'_stand']=Standardize.iterateAndStandardize(keyWord,df,trainedModelsBasicPath,50)   
Standardize.iterateAndchangeLabelToOnes(df)




################# get spacing

"""
looking through all valid spacings (if it si invalid it goes below 0)
and displaying minimal maximal and rounded mean spacing and median
in my case median and mean values are close - and using the median values will lead to a bit less interpolations later
"""
spacingDict={}
for keyWord in ['t2w','adc', 'cor','hbv','sag'  ]: 
    for addedKey in ['_spac_x','_spac_y','_spac_z']:   
        colName = keyWord+addedKey
        liist = list(filter(lambda it: it>0 ,df[colName].to_numpy() ))
        minn=np.min(liist)                
        maxx=np.max(liist)
        meanRounded = round((minn+maxx)/2,1)
        medianRounded = round(np.median(liist),1)
        spacingDict[colName]=(minn,maxx,meanRounded,medianRounded)

targetSpacingg=(spacingDict['t2w_spac_x'][3],spacingDict['t2w_spac_y'][3],spacingDict['t2w_spac_z'][3])
print("**************    target spacing    ***************")    
print(targetSpacingg)

"""
registered images were already resampled now time for t2w and labels
"""
def resample_ToMedianSpac(row,colName,targetSpacing):
    path=row[colName]
    study_id=str(row['study_id'])
    
    newPath = path.replace(".mha","_medianSpac.mha" )
    if(not pathOs.exists(newPath)):   
        experiment.log_text(f" new resample {colName} {study_id}")

        try:
            resampled = Resampling.resample_with_GAN(path,targetSpacing)
        except:
            print("error resampling")
        resampled = Resampling.resample_with_GAN(path,targetSpacing)

        write_to_modif_path(resampled,path,".mha","_medianSpac.mha" )
    else:
        experiment.log_text(f" old resample {colName} {study_id}")
        print("already resampled")
    return newPath    




def resample_labels(row,targetSpacing):
    path_t2w=row['t2w']
    path= path_t2w.replace(".mha","_stand_label.mha")
    
    study_id=str(row['study_id'])
   
    
    newPath = path.replace(".mha","_medianSpac.mha" )
    if(not pathOs.exists(newPath)):         
        try:
            experiment.log_text(f" new resample label {study_id}")
            resampled = Resampling.resample_label_with_GAN(path,targetSpacing)
        except:
            print("error resampling")
        resampled = Resampling.resample_label_with_GAN(path,targetSpacing)

        write_to_modif_path(resampled,path,".mha","_medianSpac.mha" )
    else:
        print("already resampled")
        experiment.log_text(f"already reSampled label {study_id}")
        
    
    return newPath        

#needs to be on single thread as resampling GAN is acting on GPU
# we save the metadata to main pandas data frame 
df["adc_med_spac"]=df.apply(lambda row : resample_ToMedianSpac(row, 'adc_stand',targetSpacingg)   , axis = 1) 
df["hbv_med_spac"]=df.apply(lambda row : resample_ToMedianSpac(row, 'hbv_stand',targetSpacingg)   , axis = 1) 
df["t2w_med_spac"]=df.apply(lambda row : resample_ToMedianSpac(row, 't2w_stand',targetSpacingg)   , axis = 1) 
df["label_med_spac"]=df.apply(lambda row : resample_labels(row,targetSpacingg)   , axis = 1) 


#######Registration of adc and hb
elacticPath='/home/sliceruser/Slicer/NA-MIC/Extensions-30822/SlicerElastix/lib/Slicer-5.0/elastix'
reg_prop='/home/sliceruser/data/piCaiCode/preprocessing/registration/parameters.txt'      
       
for keyWord in ['adc_med_spac','hbv_med_spac']:
    resList=[]     
    # list(map(partial(reg_adc_hbv_to_t2w,colName=keyWord,elacticPath=elacticPath,reg_prop=reg_prop,t2wColName='t2w_med_spac',experiment=experiment ),list(df.iterrows())))
    
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(partial(reg_adc_hbv_to_t2w,colName=keyWord,elacticPath=elacticPath,reg_prop=reg_prop,t2wColName='t2w_med_spac'),list(df.iterrows()))    
        df['registered_'+keyWord]=resList  





# def ifShortReturnMinus(tupl, patId,colName):
#     if(len(tupl)!=3):
#         print("incorrect spacial data "+ colName+ "  "+patId+ " length "+ len(tupl) ) 
#         return (-1,-1,-1)
#     return tupl 

# ######Now we need to retrieve the maximum dimensions of resampled images

# def get_spatial_meta(row,colName):
#     row=row[1]
#     patId=str(row['patient_id'])
#     path=str(row[colName])
#     if(len(path)>1):
#         image = sitk.ReadImage(path)
#         sizz= ifShortReturnMinus(image.GetSize(),patId,colName )
#         spac= ifShortReturnMinus(image.GetSpacing(),patId,colName)
#         orig= ifShortReturnMinus(image.GetOrigin(),patId,colName)
#         return list(sizz)+list(spac)+list(orig)
#     return [-1,-1,-1,-1,-1,-1,-1,-1,-1]

# for keyWord in ['t2w_medianSpac']:    
#     resList=[]
#     with mp.Pool(processes = mp.cpu_count()) as pool:
#         resList=pool.map(partial(get_spatial_meta,colName=keyWord)  ,list(df.iterrows()))    
#     print(type(resList))    
#     df[keyWord+'_sizz_x']= list(map(lambda arr:arr[0], resList))    
#     df[keyWord+'_sizz_y']= list(map(lambda arr:arr[1], resList))    
#     df[keyWord+'_sizz_z']= list(map(lambda arr:arr[2], resList))
    
# df.to_csv('/home/sliceruser/data/metadata/processedMetaData.csv') 

# #getting maximum size - so one can pad to uniform size if needed (for example in validetion test set)
# median_spac_max_size_x = np.max(list(filter(lambda it: it>0 ,df['t2w_med_spac_sizz_x'].to_numpy() )))
# median_spac_max_size_y = np.max(list(filter(lambda it: it>0 ,df['t2w_med_spac_sizz_y'].to_numpy() )))
# median_spac_max_size_z = np.max(list(filter(lambda it: it>0 ,df['t2w_med_spac_sizz_z'].to_numpy() )))


# maxSize = (median_spac_max_size_x,median_spac_max_size_y,median_spac_max_size_z  )
# print(maxSize)



print("fiiiniiished")
print(df['study_id'])

    
# def resample_registered_to_given(row,colname,targetSpacing):
#     path=row[colname]
#     outPath = path.replace(".mha","_for_"+colName)
#     registeredPath = outPath+"/result.0.mha"
#     newPath = path.replace(".mha","_medianSpac.mha" )   
#     try:
#         resampled = Resampling.resample_with_GAN(registeredPath,targetSpacing)
#     except:
#         print("error resampling")
#     resampled = Resampling.resample_with_GAN(registeredPath,targetSpacing)

#     write_to_modif_path(resampled,path,".mha","_medianSpac.mha" )
#     return newPath


#######Setting spacing of adc and HBV to t2w so then there would be less resampling needed during registration

# def resample_adc_hbv_to_t2w(row,secondCol ):
#     pathT2w= row['t2w']
#     pathh= row[secondCol] 
#     newPath = pathh.replace(".mha","_resmaplA.mha" )

#     imageT2W = sitk.ReadImage(pathT2w)
#     targetSpacing = imageT2W.GetSpacing()
#     try:
#         resampled = Resampling.resample_with_GAN(pathh,targetSpacing)
#     except:
#         print("error resampling")
#     resampled = Resampling.resample_with_GAN(pathh,targetSpacing)

#     write_to_modif_path(resampled,pathh,".mha","_resmaplA.mha" )
#     return newPath

# #needs to be on single thread as resampling GAN is acting on GPU
# # we save the metadata to main pandas data frame 
# df["adc_resmaplA"]=df.apply(lambda row : resample_adc_hbv_to_t2w(row, 'adc')   , axis = 1) 
# df["hbv_resmaplA"]=df.apply(lambda row : resample_adc_hbv_to_t2w(row, 'hbv')   , axis = 1) 
df.to_csv('/home/sliceruser/data/metadata/processedMetaData.csv') 
        

