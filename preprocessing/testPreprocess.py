import functools
import multiprocessing as mp
import os
import os.path
from functools import partial
from os import path as pathOs
from zipfile import BadZipFile, ZipFile
import comet_ml
import numpy as np
import pandas as pd
import SimpleITK as sitk
from comet_ml import Experiment
import ManageMetadata
import Resampling
import Standardize
import utilsPreProcessing
import semisuperPreprosess
from registration.elastixRegister import (reg_adc_hbv_to_t2w,
                                          reg_adc_hbv_to_t2w_sitk)
from utilsPreProcessing import write_to_modif_path

experiment = Experiment(
    api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
    #workspace="picai", # Optional
    project_name="picai_first_preprocessing", # Optional
    #experiment_name="baseline" # Optional
)

## some paths
elacticPath='elastix'
reg_prop='/home/sliceruser/data/piCaiCode/preprocessing/registration/parameters.txt'  
trainedModelsBasicPath='/home/sliceruser/data/preprocess/standarizationModels'


df = pd.read_csv('/home/sliceruser/data/metadata/processedMetaData.csv')
#currently We want only imagfes with associated masks
# df = df.loc[df['isAnyMissing'] ==False]
# df = df.loc[df['isAnythingInAnnotated']>0 ]
# # ignore all with deficient spacing
# for keyWord in ['t2w','adc', 'cor','hbv','sag'  ]:    
#     colName=keyWord+ "_spac_x"
#     df = df.loc[df[colName]>0 ]
# # get only complete representaions and only those with labels
# df = df.loc[df['isAnyMissing'] ==False]
# df = df.loc[df['isAnythingInAnnotated']>0 ]    
#just for testing    
#df= df.head(30)
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
    print(f" pathhhh {path} ")
    if(path!= " " and path!=""):
        study_id=str(row['study_id'])
        
        newPath = path.replace(".mha",spacing_keyword+".mha" )
        if(not pathOs.exists(newPath)):   #unhash   
        #if(True):      
            experiment.log_text(f" new resample {colName} {study_id}")
            resampled=None
            try:
                print(f"resampling colname {colName} {study_id}")
                resampled = Resampling.resample_with_GAN(path,targetSpacing)
            except Exception as e:
                print(f"error resampling {e}")
                resampled = Resampling.resample_with_GAN(path,targetSpacing)
            print(f" resempled Res colname {colName}  {study_id} size {resampled.GetSize() } spacing {resampled.GetSpacing()}  ")
            write_to_modif_path(resampled,path,".mha",spacing_keyword+".mha" )
        else:
            experiment.log_text(f" old resample {colName} {study_id}")
            print(f"already resampled {study_id} colname {colName} newPath {newPath}")
        return newPath    
    return " "


def resample_To_t2w(row,colName,spacing_keyword,t2wColName):
    path=row[colName]
    if(path!= " " and path!=""):
        study_id=str(row['study_id'])
        t2wImage=sitk.ReadImage(str(row[t2wColName]))
        targetSpacing=t2wImage.GetSpacing()
        newPath = path.replace(".mha",spacing_keyword+".mha" )
        if(not pathOs.exists(newPath)):   #unhash   
        #if(True):  
            experiment.log_text(f" new resample {colName} {study_id}")
            resampled=None
            try:
                print(f"resampling colname {colName} {study_id}")
                resampled = Resampling.resample_with_GAN(path,targetSpacing)
            except Exception as e:
                print(f"error resampling {e}")
                resampled = Resampling.resample_with_GAN(path,targetSpacing)
            print(f" resempled Res colname {colName}  {study_id} size {resampled.GetSize() } spacing {resampled.GetSpacing()} newPath {newPath}  ")
            write_to_modif_path(resampled,path,".mha",spacing_keyword+".mha" )
        else:
            experiment.log_text(f" old resample {colName} {study_id}")
            print(f"already resampled {study_id} colname {colName} newPath {newPath}")
        return newPath    
    return " "





def resample_labels(row,targetSpacing,spacing_keyword):
    """
    performs labels resampling  to the target 
    """
    path=row['reSampledPath']

    if(path!= " " and path!=""):
        path_t2w=row['t2w']

        outPath= path_t2w.replace(".mha","__label.mha")
        
        study_id=str(row['study_id'])
    
        
        newPath = outPath.replace(".mha",spacing_keyword+".nii.gz" )
        if(not pathOs.exists(newPath)):         
            try:
                experiment.log_text(f" new resample label {study_id}")
                resampled = Resampling.resample_label_with_GAN(path,targetSpacing)
            except Exception as e:
                print(f"error resampling {e}")
                #recursively apply resampling
                resample_labels(row,targetSpacing,spacing_keyword)
                #resampled = Resampling.resample_label_with_GAN(path,targetSpacing)

            write_to_modif_path(resampled,outPath,".mha",spacing_keyword+".nii.gz" )
        else:
            print("already resampled")
            experiment.log_text(f"already reSampled label {study_id}")
            
        
        return newPath  
    return " "    


def join_and_save_3Channel(row,colNameT2w,colNameAdc,colNameHbv):
    """
    join 3 images into 1 3 channel image
    """
    row=row[1]
    # print(row)
    print(str(row[colNameT2w]))
    print(str(row[colNameAdc]))
    print(str(row[colNameHbv]))
    outPath = str(row[colNameT2w]).replace('.mha', '_3Chan.mha')
    if(str(row[colNameT2w])!= " " and str(row[colNameT2w])!="" 
        and str(row[colNameAdc])!= " " and str(row[colNameAdc])!="" 
        and str(row[colNameHbv])!= " " and str(row[colNameHbv])!=""
        ):
        if(not pathOs.exists(outPath)):
            patId=str(row['patient_id'])

            imgT2w=sitk.ReadImage(str(row[colNameT2w]))
            imgAdc=sitk.ReadImage(str(row[colNameAdc]))
            imgHbv=sitk.ReadImage(str(row[colNameHbv]))

            imgT2w=sitk.Cast(imgT2w, sitk.sitkFloat32)
            imgAdc=sitk.Cast(imgAdc, sitk.sitkFloat32)
            imgHbv=sitk.Cast(imgHbv, sitk.sitkFloat32)
            print(f"patient id  {patId} ")
            print(f"t2w size {imgT2w.GetSize() } spacing {imgT2w.GetSpacing()} ")    
            print(f"adc size {imgAdc.GetSize() } spacing {imgAdc.GetSpacing()} ")    
            print(f"hbv size {imgHbv.GetSize() } spacing {imgHbv.GetSpacing()} ")    

            join = sitk.JoinSeriesImageFilter()
            joined_image = join.Execute(imgT2w, imgAdc,imgHbv)
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            writer.SetFileName(outPath)
            writer.Execute(joined_image)
        return outPath      
    return " "




def resize_and_join(row,colNameT2w,colNameAdc,colNameHbv
,sizeWord,targetSize,ToBedivisibleBy32,paddValue=0.0):
    """
    resizes and join images into 3 channel images 
    if ToBedivisibleBy32 is set to true size will be set to be divisible to 32 
    and targetSize will be ignored if will be false
    size will be padded to targetSize
    """
    #row=row[1]
    print(f"resize_and_join colNameT2w {colNameT2w} row[1] str(row[colNameT2w]) {str(row[1][colNameT2w])}  ")
    outPath = str(row[1][colNameT2w]).replace('.mha',sizeWord+ '_34Chan.mha')
    if(str(row[1][colNameT2w])!= " " and str(row[1][colNameT2w])!="" 
        and str(row[1][colNameAdc])!= " " and str(row[1][colNameAdc])!="" 
        and str(row[1][colNameHbv])!= " " and str(row[1][colNameHbv])!=""
        ):
        if(not pathOs.exists(outPath)):
            patId=str(row[1]['patient_id'])
            print(f" str(row[1][colNameAdc])  {str(row[1][colNameAdc])}  str(row[1][colNameHbv]) {str(row[1][colNameHbv])}"    )
            imgT2w=sitk.ReadImage(str(row[1][colNameT2w]))
            imgAdc=sitk.ReadImage(str(row[1][colNameAdc]))
            imgHbv=sitk.ReadImage(str(row[1][colNameHbv]))

            imgT2w=sitk.Cast(imgT2w, sitk.sitkFloat32)
            imgAdc=sitk.Cast(imgAdc, sitk.sitkFloat32)
            imgHbv=sitk.Cast(imgHbv, sitk.sitkFloat32)

            if(ToBedivisibleBy32):
                imgT2w=Standardize.padToDivisibleBy32(imgT2w,paddValue)
                imgAdc=Standardize.padToDivisibleBy32(imgAdc,paddValue)
                imgHbv=Standardize.padToDivisibleBy32(imgHbv,paddValue)
            else:
                imgT2w=Standardize.padToSize(imgT2w,targetSize,paddValue)
                imgAdc=Standardize.padToSize(imgAdc,targetSize,paddValue)
                imgHbv=Standardize.padToSize(imgHbv,targetSize,paddValue)

            print(f"patient id  {patId} ")
            print(f"t2w size {imgT2w.GetSize() } spacing {imgT2w.GetSpacing()} ")    
            print(f"adc size {imgAdc.GetSize() } spacing {imgAdc.GetSpacing()} ")    
            print(f"hbv size {imgHbv.GetSize() } spacing {imgHbv.GetSpacing()} ")    

            join = sitk.JoinSeriesImageFilter()
            joined_image = join.Execute(imgT2w, imgAdc,imgHbv)
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            writer.SetFileName(outPath)
            writer.Execute(joined_image)
        return outPath      
    return " "




    # resList=[]    
    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(join_and_save_3Channel
    #                             ,colNameT2w=t2wKeyWord
    #                             ,colNameAdc='adc'+spacing_keyword
    #                             ,colNameHbv='hbv'+spacing_keyword
    #                             )  ,list(df.iterrows())) 
    # df[t2wKeyWord+"_3Chan"+sizeWord]=resList


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
    maxSize=(maxSize[0]+1,maxSize[1]+1,maxSize[2]+1 )
    print(f" max sizee {maxSize}")
    ###now we join it into 3 channel image we save two versions one is divisible by 32 other is set to max size ...

    sizeWord="_div32_"
    resList=[]
    # resList=list(map(partial(resize_and_join
    #                             ,colNameT2w=t2wKeyWord
    #                             ,colNameAdc="adc"+spacing_keyword
    #                             ,colNameHbv="hbv"+spacing_keyword
    #                             ,sizeWord=sizeWord
    #                             ,targetSize=maxSize
    #                             ,ToBedivisibleBy32=True
    #                             )  ,list(df.iterrows())))


    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(partial(resize_and_join
                                ,colNameT2w=t2wKeyWord
                                ,colNameAdc="adc"+spacing_keyword
                                ,colNameHbv="hbv"+spacing_keyword
                                ,sizeWord=sizeWord
                                ,targetSize=maxSize
                                ,ToBedivisibleBy32=True
                                )  ,list(df.iterrows())) 
    df[t2wKeyWord+"_3Chan"+sizeWord]=resList
    #setting padding to labels
    Standardize.iterateAndpadLabels(df,"label"+spacing_keyword,maxSize, 0.0,spacing_keyword+sizeWord,True)


    # sizeWord="_maxSize_"
    # resList=[]
    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(resize_and_join
    #                             ,colNameT2w=t2wKeyWord
    #                             ,colNameAdc="adc"+spacing_keyword
    #                             ,colNameHbv="hbv"+spacing_keyword
    #                             ,sizeWord=sizeWord
    #                             ,targetSize=maxSize
    #                             ,ToBedivisibleBy32=False
    #                             )  ,list(df.iterrows())) 


    # df[t2wKeyWord+"_3Chan"+sizeWord]=resList
    # Standardize.iterateAndpadLabels(df,"label"+spacing_keyword,maxSize, 0.0,spacing_keyword+sizeWord,False)






# some preprocessing common for all



#bias field correction
# Standardize.iterateAndBiasCorrect('t2w',df)
# #Standarization
# for keyWord in ['bfc_'+'t2w','adc', 'hbv']: #'cor',,'sag'
#     ## denoising
#     #Standardize.iterateAndDenoise(keyWord,df)
#     ## standarization
#     Standardize.iterateAndStandardize(keyWord,df,trainedModelsBasicPath,80)   
# #standardize labels
# Standardize.iterateAndchangeLabelToOnes(df)

#### 
#first get adc and tbv to t2w spacing
spacing_keyword='_tw_d_'
# df["adc_d"+spacing_keyword]=df.apply(lambda row : resample_To_t2w(row,'adc','tw','t2w')   , axis = 1) 
# df["hbv_d"+spacing_keyword]=df.apply(lambda row : resample_To_t2w(row,'hbv','tw','t2w')   , axis = 1) 

#now registration of adc and hbv to t2w
for keyWord in ['adc','hbv']:
    resList=[]     
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(partial(reg_adc_hbv_to_t2w,colName=keyWord,elacticPath=elacticPath,reg_prop=reg_prop,t2wColName='t2w'),list(df.iterrows()))    

    df['registered_'+keyWord]=resList      
    # pathss = list(map(lambda tupl :tupl[0],resList   ))
    # reg_values = list(map(lambda tupl :tupl[1],resList   ))

    # df['registered_'+keyWord]=pathss  
    # df['registered_'+keyWord+"score"]=reg_values  
#adding data about number of lesions that algorithm should detect
df=semisuperPreprosess.iterate_and_addLesionNumber(df)

#checking registration by reading from logs the metrics so we will get idea how well it went


#######      
targetSpacinggg=(spacingDict['t2w_spac_x'][3],spacingDict['t2w_spac_y'][3],spacingDict['t2w_spac_z'][3])
preprocess_diffrent_spacings(df,targetSpacinggg,"_med_spac_b")
preprocess_diffrent_spacings(df,(1.0,1.0,1.0),"_one_spac_c")
# preprocess_diffrent_spacings(df,(1.5,1.5,1.5),"_one_and_half_spac_b")
# preprocess_diffrent_spacings(df,(2.0,2.0,2.0),"_two_spac_b")



print("fiiiniiished")
df.to_csv('/home/sliceruser/data/metadata/processedMetaData_current_b.csv') 
print(df['num_lesions_to_retain'])



# /home/sliceruser/data/orig/10032/10032_1000032_adc_med_spac_for_adc_med_spac/result.0.mha
# /home/sliceruser/data/orig/10032/10032_1000032_hbv_med_spac_for_hbv_med_spac/result.0.mha
    
# orig/10012/10012_1000012_adc_med_spac_for_adc_med_spac/result.0.mha
# orig/10012/10012_1000012_hbv_med_spac_for_hbv_med_spac/result.0.mha







