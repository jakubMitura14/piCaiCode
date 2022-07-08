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
print(df)    
#just for testing    
#df= df.head(4)
##df.to_csv('/home/sliceruser/data/metadata/processedMetaData_current.csv') 

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





def resample_labels(row,targetSpacing,spacing_keyword):
    path_t2w=row['t2w']
    path= path_t2w.replace(".mha","_stand_label.mha")
    
    study_id=str(row['study_id'])
   
    
    newPath = path.replace(".mha",spacing_keyword+".mha" )
    if(not pathOs.exists(newPath)):         
        try:
            experiment.log_text(f" new resample label {study_id}")
            resampled = Resampling.resample_label_with_GAN(path,targetSpacing)
        except:
            print("error resampling")
        resampled = Resampling.resample_label_with_GAN(path,targetSpacing)

        write_to_modif_path(resampled,path,".mha",spacing_keyword+".mha" )
    else:
        print("already resampled")
        experiment.log_text(f"already reSampled label {study_id}")
        
    
    return newPath  


def ifShortReturnMinus(tupl, patId,colName):
    if(len(tupl)!=3):
        print("incorrect spacial data "+ colName+ "  "+patId+ " length "+ len(tupl) ) 
        return (-1,-1,-1)
    return tupl 

def get_spatial_meta(row,colName):
    row=row[1]
    patId=str(row['patient_id'])
    path=str(row[colName])
    if(len(path)>1):
        image = sitk.ReadImage(path)
        sizz= ifShortReturnMinus(image.GetSize(),patId,colName )
        spac= ifShortReturnMinus(image.GetSpacing(),patId,colName)
        orig= ifShortReturnMinus(image.GetOrigin(),patId,colName)
        return list(sizz)+list(spac)+list(orig)
    return [-1,-1,-1,-1,-1,-1,-1,-1,-1]


def join_and_save_3Channel(row,colNameT2w,colNameAdc,colNameHbv,outPath):
    """
    join 3 images into 1 3 channel image
    """
    row=row[1]
    #patId=str(row['patient_id'])
    imgT2w=sitk.ReadImage(str(row[colNameT2w]))
    imgAdc=sitk.ReadImage(str(row[colNameAdc]))
    imgHbv=sitk.ReadImage(str(row[colNameHbv]))
    join = sitk.JoinSeriesImageFilter()
    joined_image = join.Execute(imgT2w, imgAdc,imgHbv)
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(outPath)
    writer.Execute(joined_image)      


def preprocess_diffrent_spacings(df,targetSpacingg,spacing_keyword):
    print(f"**************    target spacing    ***************  {targetSpacingg}   {spacing_keyword}")  
    ## bias field correction
    Standardize.iterateAndBiasCorrect('t2w',df)
    ## Standarization
    for keyWord in ['t2w','adc', 'cor','hbv','sag'  ]:
        ## denoising
        Standardize.iterateAndDenoise(keyWord,df)
        ## standarization
        Standardize.iterateAndStandardize(keyWord,df,trainedModelsBasicPath,50)   
    #standardize labels
    Standardize.iterateAndchangeLabelToOnes(df)



    resList=[]
    t2wKeyWord ="t2w"+spacing_keyword    

    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(partial(get_spatial_meta,colName=keyWord)  ,list(df.iterrows()))    
    print(type(resList))    
    df[t2wKeyWord+'_sizz_x']= list(map(lambda arr:arr[0], resList))    
    df[t2wKeyWord+'_sizz_y']= list(map(lambda arr:arr[1], resList))    
    df[t2wKeyWord+'_sizz_z']= list(map(lambda arr:arr[2], resList))


    #needs to be on single thread as resampling GAN is acting on GPU
    # we save the metadata to main pandas data frame 
    df["adc"+spacing_keyword]=df.apply(lambda row : resample_ToMedianSpac(row, 'adc',targetSpacingg,spacing_keyword)   , axis = 1) 
    df["hbv"+spacing_keyword]=df.apply(lambda row : resample_ToMedianSpac(row, 'hbv',targetSpacingg,spacing_keyword)   , axis = 1) 
    df["t2w"+spacing_keyword]=df.apply(lambda row : resample_ToMedianSpac(row, 't2w',targetSpacingg,spacing_keyword)   , axis = 1) 
    df["label"+spacing_keyword]=df.apply(lambda row : resample_labels(row,targetSpacingg,spacing_keyword)   , axis = 1) 


    #######Registration of adc and hb         
    for keyWord in ['adc'+spacing_keyword,'hbv'+spacing_keyword]:
        resList=[]     
        # list(map(partial(reg_adc_hbv_to_t2w,colName=keyWord,elacticPath=elacticPath,reg_prop=reg_prop,t2wColName='t2w_med_spac',experiment=experiment ),list(df.iterrows())))  
        with mp.Pool(processes = mp.cpu_count()) as pool:
            resList=pool.map(partial(reg_adc_hbv_to_t2w,colName=keyWord,elacticPath=elacticPath,reg_prop=reg_prop,t2wColName='t2w_med_spac'),list(df.iterrows()))    
            df['registered_'+keyWord]=resList  

    ManageMetadata.addSizeMetaDataToDf(t2wKeyWord,df)

    ######Now we need to retrieve the maximum dimensions of resampled images


    #getting maximum size - so one can pad to uniform size if needed (for example in validetion test set)
    median_spac_max_size_x = np.max(list(filter(lambda it: it>0 ,df[t2wKeyWord+"_sizz_x"].to_numpy() )))
    median_spac_max_size_y = np.max(list(filter(lambda it: it>0 ,df[t2wKeyWord+"_sizz_y"].to_numpy() )))
    median_spac_max_size_z = np.max(list(filter(lambda it: it>0 ,df[t2wKeyWord+"_sizz_z"].to_numpy() )))
    maxSize = (median_spac_max_size_x,median_spac_max_size_y,median_spac_max_size_z  )

    print(f" max sizee {maxSize}")


    ###now we join it into 3 channel image
    # resList=[]
    # outPath = t2wKeyWord.replace('.mha', '_3Chan.mha')

    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(join_and_save_3Channel
    #                             ,t2wKeyWord+spacing_keyword
    #                             ,colNameAdc='registered_'+'adc'+spacing_keyword
    #                             ,colNameHbv='registered_'+'hbv'+spacing_keyword
    #                             ,outPath=outPath
    #                             )  ,list(df.iterrows())) 
    # df['registered_'+keyWord+"_3Chan"]=resList

##### 
targetSpacinggg=(spacingDict['t2w_spac_x'][3],spacingDict['t2w_spac_y'][3],spacingDict['t2w_spac_z'][3])
preprocess_diffrent_spacings(df,targetSpacinggg,"_med_spac")
preprocess_diffrent_spacings(df,(1.0,1.0,1.0),"_one_spac")
preprocess_diffrent_spacings(df,(1.5,1.5,1.5),"_one_and_half_spac")
preprocess_diffrent_spacings(df,(2.0,2.0,2.0),"_two_spac")


print("fiiiniiished")
print(df['study_id'])
df.to_csv('/home/sliceruser/data/metadata/processedMetaData_current.csv') 

    
