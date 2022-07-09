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
from registration.elastixRegister import reg_adc_hbv_to_t2w,reg_adc_hbv_to_t2w_sitk
import os.path
from os import path as pathOs
import comet_ml
from comet_ml import Experiment
import ManageMetadata

experiment = Experiment(
    api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
    #workspace="picai", # Optional
    project_name="picai_first_preprocessing", # Optional
    #experiment_name="baseline" # Optional
)

## some paths
elacticPath='/home/sliceruser/Slicer/NA-MIC/Extensions-30822/SlicerElastix/lib/Slicer-5.0/elastix'
reg_prop='/home/sliceruser/data/piCaiCode/preprocessing/registration/parameters.txt'  
trainedModelsBasicPath='/home/sliceruser/data/preprocess/standarizationModels'


df = pd.read_csv('/home/sliceruser/data/metadata/processedMetaData.csv')
#currently We want only imagfes with associated masks
df = df.loc[df['isAnyMissing'] ==False]
df = df.loc[df['isAnythingInAnnotated']>0 ]
# ignore all with deficient spacing
for keyWord in ['t2w','adc', 'cor','hbv','sag'  ]:    
    colName=keyWord+ "_spac_x"
    df = df.loc[df[colName]>0 ]
# get only complete representaions and only those with labels
df = df.loc[df['isAnyMissing'] ==False]
df = df.loc[df['isAnythingInAnnotated']>0 ]    
#just for testing    
#df= df.head(12)
##df.to_csv('/home/sliceruser/data/metadata/processedMetaData_current.csv') 
print(df)    

########## Standarization




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


"""
registered images were already resampled now time for t2w and labels
"""
def resample_ToMedianSpac(row,colName,targetSpacing,spacing_keyword):
    path=row[colName]
    if(path!= " " and path!=""):
        study_id=str(row['study_id'])
        
        newPath = path.replace(".mha",spacing_keyword+".mha" )
        if(not pathOs.exists(newPath)):      
            experiment.log_text(f" new resample {colName} {study_id}")

            try:
                resampled = Resampling.resample_with_GAN(path,targetSpacing)
            except:
                print("error resampling")
            resampled = Resampling.resample_with_GAN(path,targetSpacing)

            write_to_modif_path(resampled,path,".mha",spacing_keyword+".mha" )
        else:
            experiment.log_text(f" old resample {colName} {study_id}")
            print("already resampled")
        return newPath    
    return " "




def resample_labels(row,targetSpacing,spacing_keyword):
    """
    performs labels resampling  to the target 
    """
    path=row['reSampledPath']

    if(path!= " " and path!=""):
        path_t2w=row['t2w']

        outPath= path_t2w.replace(".mha","_stand_label.mha")
        
        study_id=str(row['study_id'])
    
        
        newPath = path.replace(".mha",spacing_keyword+".mha" )
        if(not pathOs.exists(newPath)):         
            try:
                experiment.log_text(f" new resample label {study_id}")
                resampled = Resampling.resample_label_with_GAN(path,targetSpacing)
            except:
                print("error resampling")
                resampled = Resampling.resample_label_with_GAN(path,targetSpacing)

            write_to_modif_path(resampled,outPath,".mha",spacing_keyword+".mha" )
        else:
            print("already resampled")
            experiment.log_text(f"already reSampled label {study_id}")
            
        
        return newPath  
    return " "    


def join_and_save_3Channel(row,colNameT2w,colNameAdc,colNameHbv,outPath):
    """
    join 3 images into 1 3 channel image
    """
    row=row[1]
    print("debug")
    print(row)
    print(str(row[colNameT2w]))
    print(str(row[colNameAdc]))
    print(str(row[colNameHbv]))
    if(str(row[colNameT2w])!= " " and str(row[colNameT2w])!="" 
        and str(row[colNameAdc])!= " " and str(row[colNameAdc])!="" 
        and str(row[colNameHbv])!= " " and str(row[colNameHbv])!=""):
        #patId=str(row['patient_id'])

        imgT2w=sitk.ReadImage(str(row[colNameT2w]))
        imgAdc=sitk.ReadImage(str(row[colNameAdc]))
        imgHbv=sitk.ReadImage(str(row[colNameHbv]))

        imgT2w=sitk.Cast(imgT2w, sitk.sitkFloat32)
        imgAdc=sitk.Cast(imgAdc, sitk.sitkFloat32)
        imgHbv=sitk.Cast(imgHbv, sitk.sitkFloat32)

        join = sitk.JoinSeriesImageFilter()
        joined_image = join.Execute(imgT2w, imgAdc,imgHbv)
        writer = sitk.ImageFileWriter()
        writer.KeepOriginalImageUIDOn()
        writer.SetFileName(outPath)
        writer.Execute(joined_image)
        return outPath      
    return " "

#### 



def preprocess_diffrent_spacings(df,targetSpacingg,spacing_keyword):
    print(f"**************    target spacing    ***************  {targetSpacingg}   {spacing_keyword}")  
    t2wKeyWord ="t2w"+spacing_keyword    
    #needs to be on single thread as resampling GAN is acting on GPU
    # we save the metadata to main pandas data frame 
    df["adc"+spacing_keyword]=df.apply(lambda row : resample_ToMedianSpac(row, 'registered_'+'adc',targetSpacingg,spacing_keyword)   , axis = 1) 
    df["hbv"+spacing_keyword]=df.apply(lambda row : resample_ToMedianSpac(row, 'registered_'+'hbv',targetSpacingg,spacing_keyword)   , axis = 1) 
    df["t2w"+spacing_keyword]=df.apply(lambda row : resample_ToMedianSpac(row, 't2w',targetSpacingg,spacing_keyword)   , axis = 1) 
    df["label"+spacing_keyword]=df.apply(lambda row : resample_labels(row,targetSpacingg,spacing_keyword)   , axis = 1) 



    ManageMetadata.addSizeMetaDataToDf(t2wKeyWord,df)

    ######Now we need to retrieve the maximum dimensions of resampled images

    #getting maximum size - so one can pad to uniform size if needed (for example in validetion test set)
    maxSize = ManageMetadata.getMaxSize(t2wKeyWord,df)

    print(f" max sizee {maxSize}")
    ###now we join it into 3 channel image
    resList=[]
    outPath = t2wKeyWord.replace('.mha', '_3Chan.mha')



    resList=list(map(partial(join_and_save_3Channel
                                ,colNameT2w=t2wKeyWord
                                ,colNameAdc='adc'+spacing_keyword
                                ,colNameHbv='hbv'+spacing_keyword
                                ,outPath=outPath
                                )  ,list(df.iterrows())))

    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(join_and_save_3Channel
    #                             ,colNameT2w=t2wKeyWord
    #                             ,colNameAdc='registered_'+'adc'+spacing_keyword
    #                             ,colNameHbv='registered_'+'hbv'+spacing_keyword
    #                             ,outPath=outPath
    #                             )  ,list(df.iterrows())) 
    df[t2wKeyWord+"_3Chan"]=resList


# some preprocessing common for all
# bias field correction

Standardize.iterateAndBiasCorrect('t2w',df)
#Standarization
for keyWord in ['t2w','adc', 'hbv'  ]: #'cor',,'sag'
    ## denoising
    #Standardize.iterateAndDenoise(keyWord,df)
    ## standarization
    Standardize.iterateAndStandardize(keyWord,df,trainedModelsBasicPath,50)   
#standardize labels
Standardize.iterateAndchangeLabelToOnes(df)



# #######Registration of adc and hb         
# for keyWord in ['adc','hbv']:
#     resList=[]     
#     # list(map(partial(reg_adc_hbv_to_t2w,colName=keyWord,elacticPath=elacticPath,reg_prop=reg_prop,t2wColName='t2w_med_spac',experiment=experiment ),list(df.iterrows())))  
#     # resList=list(map(partial(reg_adc_hbv_to_t2w,colName=keyWord,elacticPath=elacticPath,reg_prop=reg_prop,t2wColName='t2w'),list(df.iterrows())) )   
#     # df['registered_'+keyWord]=resList  
    
#     # with mp.Pool(processes = mp.cpu_count()) as pool:
#     #     resList=pool.map(partial(reg_adc_hbv_to_t2w,colName=keyWord,elacticPath=elacticPath,reg_prop=reg_prop,t2wColName='t2w'),list(df.iterrows()))    
#     # df['registered_'+keyWord]=resList  
#     with mp.Pool(processes = mp.cpu_count()) as pool:
#         resList=pool.map(partial(reg_adc_hbv_to_t2w_sitk
#                 ,colName=keyWord
#                 ,t2wColName='t2w'),list(df.iterrows()))    
#     df['registered_'+keyWord]=resList  


# #reg_adc_hbv_to_t2w_sitk(row,colName,t2wColName)
# #####
 
# targetSpacinggg=(spacingDict['t2w_spac_x'][3],spacingDict['t2w_spac_y'][3],spacingDict['t2w_spac_z'][3])
# preprocess_diffrent_spacings(df,targetSpacinggg,"_med_spac")
# preprocess_diffrent_spacings(df,(1.0,1.0,1.0),"_one_spac")
# preprocess_diffrent_spacings(df,(1.5,1.5,1.5),"_one_and_half_spac")
# preprocess_diffrent_spacings(df,(2.0,2.0,2.0),"_two_spac")


print("fiiiniiished")
print(df['study_id'])
df.to_csv('/home/sliceruser/data/metadata/processedMetaData_current.csv') 



# /home/sliceruser/data/orig/10032/10032_1000032_adc_med_spac_for_adc_med_spac/result.0.mha
# /home/sliceruser/data/orig/10032/10032_1000032_hbv_med_spac_for_hbv_med_spac/result.0.mha
    
# orig/10012/10012_1000012_adc_med_spac_for_adc_med_spac/result.0.mha
# orig/10012/10012_1000012_hbv_med_spac_for_hbv_med_spac/result.0.mha







