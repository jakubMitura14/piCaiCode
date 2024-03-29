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
import math
import dask
import dask.dataframe as dd

experiment = Experiment(
    api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
    #workspace="picai", # Optional
    project_name="picai_first_preprocessing", # Optional
    #experiment_name="baseline" # Optional
)

## some paths
elacticPath='/home/sliceruser/elastixBase/elastix-5.0.1-linux/bin/elastix'
reg_prop='/home/sliceruser/data/piCaiCode/preprocessing/registration/parameters.txt'  
trainedModelsBasicPath='/home/sliceruser/data/preprocess/standarizationModels'

physical_size =(81.0, 160.0, 192.0)#taken from picai used to crop image so only center will remain


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
#df= df.head(60)
##df.to_csv('/home/sliceruser/data/metadata/processedMetaData_current.csv') 
print(df)    







################# get spacing
"""
looking through all valid spacings (if it si invalid it goes below 0)
and displaying minimal maximal and rounded mean spacing and median
in my case median and mean values are close - and using the median values will lead to a bit less interpolations later
"""

# spacingDict={}
# for keyWord in ['t2w','adc', 'cor','hbv','sag'  ]: 
#     for addedKey in ['_spac_x','_spac_y','_spac_z']:   
#         colName = keyWord+addedKey
#         liist = list(filter(lambda it: it>0 ,df[colName].to_numpy() ))
#         minn=np.min(liist)                
#         maxx=np.max(liist)
#         meanRounded = round((minn+maxx)/2,1)
#         medianRounded = round(np.median(liist),1)
#         spacingDict[colName]=(minn,maxx,meanRounded,medianRounded)


"""
registered images were already resampled now time for t2w and labels
"""
def resample_ToMedianSpac(row,colName,targetSpacing,spacing_keyword):
    #row=row[1]    
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
        #if(not pathOs.exists(newPath)):   #unhash   
        if(True):  
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
    #row=row[1]    
    path=row['label_stand']

    print(f"lllll lab path {path} t2wPath {row['t2w']}")


    if(path!= " " and path!=""):
        path_t2w=row['t2w']

        outPath= path_t2w.replace(".mha","__label.mha")
        
        study_id=str(row['study_id'])
    
        
        newPath = outPath.replace(".mha",spacing_keyword+"lab_.nii.gz" )
        if(not pathOs.exists(newPath) and 'nii' in path ):         
        #if(True):         
            # try:
            print(" resampling label A ")
            #experiment.log_text(f" new resample label {study_id}")
            resampled = Resampling.resample_label_with_GAN(path,targetSpacing)
            # except Exception as e:
            #     print(f"error resampling {e}")
            #     #recursively apply resampling
            #     resample_labels(row,targetSpacing,spacing_keyword)
            #     #resampled = Resampling.resample_label_with_GAN(path,targetSpacing)
            writer = sitk.ImageFileWriter()
            writer.KeepOriginalImageUIDOn()
            writer.SetFileName(newPath)
            writer.Execute(resampled)

            print(f"in resample labels  newPath {newPath} old path {path} ")
            
            return newPath  
        else:
            print("already resampled")
            experiment.log_text(f"already reSampled label {study_id}")
            return newPath  
    return " "    




def resize_and_join(row,colNameT2w,colNameAdc,colNameHbv
,sizeWord,targetSize,ToBedivisibleBy32,labelColName,tuplNum,paddValue=0.0):
    """
    resizes and join images into 3 channel images 
    if ToBedivisibleBy32 is set to true size will be set to be divisible to 32 
    and targetSize will be ignored if will be false
    size will be padded to targetSize
    """
    ##row=row[1]
    print(f"resize_and_join labelColName {labelColName} labb {str(row[labelColName]) !=' '} source {str(row[labelColName])} ")
    outPath = str(row[colNameT2w]).replace('.mha',sizeWord+ '_34Chan_ee.nii.gz')
    outLabelPath=str(row[labelColName]).replace('.nii.gz',sizeWord+ 'labeee.nii.gz')
    outt2wPath=str(row[colNameT2w]).replace('.mha',sizeWord+ 't2weee.nii.gz')
    outadcPath=str(row[colNameAdc]).replace('.mha',sizeWord+ 'adceee.nii.gz')
    outhbvPath=str(row[colNameHbv]).replace('.mha',sizeWord+ 'hbveee.nii.gz')
    
    outLabelPathBool=False
    outt2wPathBool=False
    outadcPathBool=False
    outhbvPathBool=False
    

    if(str(row[colNameT2w])!= " " and str(row[colNameT2w])!="" and len(str(row[colNameT2w]))>5 
        and str(row[colNameAdc])!= " " and str(row[colNameAdc])!=""  and len(str(row[colNameAdc]))>5 
        and str(row[colNameHbv])!= " " and str(row[colNameHbv])!="" and len(str(row[colNameHbv]))>5 
        ):
        outt2wPathBool=True
        outadcPathBool=True
        outhbvPathBool=True
        if(not pathOs.exists(outt2wPath)):
        #if(True):
            # print(f" pathDebugT2w {pathDebugT2w} outLabelPath {outLabelPath} ")
            patId=str(row['patient_id'])
            # print(f" {str(row[colNameAdc])}  str(row[colNameHbv]) {str(row[colNameHbv])}"    )        
            
            imgT2w=sitk.ReadImage(str(row[colNameT2w]))
            imgAdc=sitk.ReadImage(str(row[colNameAdc]))
            imgHbv=sitk.ReadImage(str(row[colNameHbv]))
            

            imgT2w=sitk.Cast(imgT2w, sitk.sitkFloat32)
            imgAdc=sitk.Cast(imgAdc, sitk.sitkFloat32)
            imgHbv=sitk.Cast(imgHbv, sitk.sitkFloat32)

            imgT2w=Standardize.padToSize(imgT2w,targetSize,paddValue)
            imgAdc=Standardize.padToSize(imgAdc,targetSize,paddValue)
            imgHbv=Standardize.padToSize(imgHbv,targetSize,paddValue)


            writer = sitk.ImageFileWriter()
            writer.SetFileName(outt2wPath)
            writer.Execute(imgT2w)

            writer = sitk.ImageFileWriter()
            writer.SetFileName(outhbvPath)
            writer.Execute(imgHbv)

            writer = sitk.ImageFileWriter()
            writer.SetFileName(outadcPath)
            writer.Execute(imgAdc)

  
    if(str(row[labelColName])!= " " and str(row[labelColName])!="" and len(str(row[labelColName]))>5 ):
        outLabelPathBool=True
        imgLabel=sitk.ReadImage(str(row[labelColName]))
        imgLabel=sitk.Cast(imgLabel, sitk.sitkUInt8)
        imgLabel=Standardize.padToSize(imgLabel,targetSize,paddValue)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(outLabelPath)
        writer.Execute(imgLabel)     
        # if os.path.exists(row[labelColName]):
        #     os.remove(row[labelColName])            
            # print(f"pre patient id  {patId} ")
            # print(f"pre t2w size {imgT2w.GetSize() } spacing {imgT2w.GetSpacing()} ")    
            # print(f"pre adc size {imgAdc.GetSize() } spacing {imgAdc.GetSpacing()} ")    
            # print(f"pre hbv size {imgHbv.GetSize() } spacing {imgHbv.GetSpacing()} ")    
            # print(f"pre imgLabel size {imgLabel.GetSize() } spacing {imgLabel.GetSpacing()} ")    

            # join = sitk.JoinSeriesImageFilter()
            # joined_image = join.Execute(imgT2w, imgHbv,imgAdc,imgHbv)
            # joined_image=Standardize.padToSize(joined_image,targetSize,paddValue)

            # writer = sitk.ImageFileWriter()
            # writer.SetFileName(outPath)
            # writer.Execute(joined_image)

            # select = sitk.VectorIndexSelectionCastImageFilter()
            # imgT2w = select.Execute(joined_image, 0, sitk.sitkFloat32)
            # imgAdc = select.Execute(joined_image, 1, sitk.sitkFloat32)
            # imgHbv = select.Execute(joined_image, 2, sitk.sitkFloat32)
            # imgLabel = select.Execute(joined_image, 3, sitk.sitkUInt8)

            # join = sitk.JoinSeriesImageFilter()
            # joined_image = join.Execute(imgT2w,imgAdc, imgHbv,imgAdc)

            # if(ToBedivisibleBy32):
            #     imgT2w=Standardize.padToDivisibleBy32(imgT2w,paddValue)
            #     imgAdc=Standardize.padToDivisibleBy32(imgAdc,paddValue)
            #     imgHbv=Standardize.padToDivisibleBy32(imgHbv,paddValue)
            #     imgLabel=Standardize.padToDivisibleBy32(imgLabel,paddValue)
            # else:
            #     imgT2w=Standardize.padToSize(imgT2w,targetSize,paddValue)
            #     imgAdc=Standardize.padToSize(imgAdc,targetSize,paddValue)
            #     imgHbv=Standardize.padToSize(imgHbv,targetSize,paddValue)
            #     imgLabel=Standardize.padToSize(imgLabel,targetSize,paddValue)

            # print(f"post patient id  {patId} ")
            # print(f"post t2w size {imgT2w.GetSize() } spacing {imgT2w.GetSpacing()} ")    
            # print(f"post adc size {imgAdc.GetSize() } spacing {imgAdc.GetSpacing()} ")    
            # print(f"post hbv size {imgHbv.GetSize() } spacing {imgHbv.GetSpacing()} ")    
            # print(f"post imgLabel size {imgLabel.GetSize() } spacing {imgLabel.GetSpacing()} ")    
    if (os.path.exists(row[colNameT2w]) and outt2wPathBool):
        os.remove(row[colNameT2w])
    if (os.path.exists(row[colNameAdc]) and outadcPathBool ):
        os.remove(row[colNameAdc])
    if (os.path.exists(row[colNameHbv]) and outhbvPathBool):
        os.remove(row[colNameHbv])


            
    if(not outLabelPathBool):
        outLabelPath=" "
    if(not outt2wPathBool):
        outt2wPath=" "
    if(not outadcPathBool):
        outadcPath=" "
    if(not outhbvPathBool):
        outhbvPath=" "




    return (outLabelPath,outt2wPath,outadcPath,outhbvPath)[tuplNum]
                   




    # resList=[]    
    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(join_and_save_3Channel
    #                             ,colNameT2w=t2wKeyWord
    #                             ,colNameAdc='adc'+spacing_keyword
    #                             ,colNameHbv='hbv'+spacing_keyword
    #                             )  ,list(df.iterrows())) 
    # df[t2wKeyWord+"_3Chan"+sizeWord]=resList

def getDummy(df,spac):
    df= df.compute()
    label_name=f"label_{spac}" 
    print(df[label_name])
    imageRef_path=list(filter(lambda it: it!= " ", df[label_name].to_numpy()))[0]
    dummyLabelPath=f"/home/sliceruser/data/dummyData/zeroLabel{spac}.nii.gz"
    sizz=semisuperPreprosess.writeDummyLabels(dummyLabelPath,imageRef_path)
    img_size = (sizz[2],sizz[1],sizz[0])
    return(dummyLabelPath,img_size)

def addDummyLabelPath(row, labelName, dummyLabelPath):
    """
    adds dummy label to the given column in every spot it is empty
    """
    # row = row[1]
    if(row[labelName]==' '):
        return dummyLabelPath
    else:
        return row[labelName]    

def preprocess_diffrent_spacings(df,targetSpacingg,spacing_keyword):
    print(f"**************    target spacing    ***************  {targetSpacingg}   {spacing_keyword}")  
    t2wKeyWord ="t2w"+spacing_keyword    
    #needs to be on single thread as resampling GAN is acting on GPU
    # we save the metadata to main pandas data frame 

    labelColName="label"+spacing_keyword


    df["adc"+spacing_keyword]=   df.apply(partial(resample_ToMedianSpac,colName='registered_'+'adc',targetSpacing=targetSpacingg, spacing_keyword=spacing_keyword  ), axis=1).compute()
    df["hbv"+spacing_keyword]=   df.apply(partial(resample_ToMedianSpac,colName='registered_'+'hbv',targetSpacing=targetSpacingg, spacing_keyword=spacing_keyword  ), axis=1).compute()
    df["t2w"+spacing_keyword]=   df.apply(partial(resample_ToMedianSpac,colName='t2w',targetSpacing=targetSpacingg, spacing_keyword=spacing_keyword  ), axis=1).compute()
    df[labelColName]=   df.apply(partial(resample_labels,targetSpacing=targetSpacingg, spacing_keyword=spacing_keyword  ), axis=1).compute()
    

    
    # resList=[]
    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(resample_ToMedianSpac,colName='registered_'+'adc',targetSpacing=targetSpacingg, spacing_keyword=spacing_keyword  ),list(df.iterrows()))    
    # df["adc"+spacing_keyword]=resList


    # resList=[]
    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(resample_ToMedianSpac,colName='registered_'+'hbv',targetSpacing=targetSpacingg, spacing_keyword=spacing_keyword  ),list(df.iterrows()))    
    # df["hbv"+spacing_keyword]=resList

    # resList=[]
    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(resample_ToMedianSpac,colName='t2w',targetSpacing=targetSpacingg, spacing_keyword=spacing_keyword  ),list(df.iterrows()))    
    # df["t2w"+spacing_keyword]=resList

    # resList=[]
    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(resample_labels,targetSpacing=targetSpacingg, spacing_keyword=spacing_keyword  ),list(df.iterrows()))    
    # df[labelColName]=resList

    # df["adc"+spacing_keyword]=df.apply(lambda row : resample_ToMedianSpac(row, 'registered_'+'adc',targetSpacingg,spacing_keyword)   , axis = 1) 
    # df["hbv"+spacing_keyword]=df.apply(lambda row : resample_ToMedianSpac(row, 'registered_'+'hbv',targetSpacingg,spacing_keyword)   , axis = 1) 
    # df["t2w"+spacing_keyword]=df.apply(lambda row : resample_ToMedianSpac(row, 't2w',targetSpacingg,spacing_keyword)   , axis = 1) 
    # df["label"+spacing_keyword]=df.apply(lambda row : resample_labels(row,targetSpacingg,spacing_keyword)   , axis = 1) 



    # ManageMetadata.addSizeMetaDataToDf(t2wKeyWord,df)

    ######Now we need to retrieve the maximum dimensions of resampled images

    #getting maximum size - so one can pad to uniform size if needed (for example in validetion test set)
    # maxSize = ManageMetadata.getMaxSize(t2wKeyWord,df)
    # maxSize=(maxSize[0]+1,maxSize[1]+1,maxSize[2]+1 )
    # print(f" max sizee {maxSize}")
    ###now we join it into 3 channel image we save two versions one is divisible by 32 other is set to max size ...

    # sizeWord="_div32_"
    # resList=[]
    # # resList=list(map(partial(resize_and_join
    # #                             ,colNameT2w=t2wKeyWord
    # #                             ,colNameAdc="adc"+spacing_keyword
    # #                             ,colNameHbv="hbv"+spacing_keyword
    # #                             ,sizeWord=sizeWord
    # #                             ,targetSize=maxSize
    # #                             ,ToBedivisibleBy32=True
    # #                             )  ,list(df.iterrows())))


    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(resize_and_join
    #                             ,colNameT2w=t2wKeyWord
    #                             ,colNameAdc="adc"+spacing_keyword
    #                             ,colNameHbv="hbv"+spacing_keyword
    #                             ,sizeWord=sizeWord
    #                             ,targetSize=maxSize
    #                             ,ToBedivisibleBy32=True
    #                             ,targetSpacingg=targetSpacingg
    #                             )  ,list(df.iterrows())) 
    # df[t2wKeyWord+"_3Chan"+sizeWord]=resList
    #setting padding to labels
    # df= Standardize.iterateAndpadLabels(df,"label"+spacing_keyword,(2,2,2), 0.0,spacing_keyword,True)
    sizzX= physical_size[0]/targetSpacingg[0]
    sizzY= physical_size[1]/targetSpacingg[1]
    sizzZ= physical_size[2]/targetSpacingg[2]
    sizz=(sizzX,sizzY,sizzZ)
    multNum=32#32
    targetSize=(math.ceil(sizz[0]/multNum)*multNum, math.ceil(sizz[1]/multNum)*multNum,math.ceil(sizz[2]/multNum)*multNum  )
    
    sizeWord="_maxSize_"
    resList=[]




    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(resize_and_join
    #                             ,colNameT2w=t2wKeyWord
    #                             ,colNameAdc="adc"+spacing_keyword
    #                             ,colNameHbv="hbv"+spacing_keyword
    #                             ,sizeWord=""
    #                             ,targetSize=targetSize
    #                             ,ToBedivisibleBy32=False
    #                             ,labelColName=labelColName
    #                             )  ,list(df.iterrows())) 
    # resList=list(map(partial(resize_and_join
    #                         ,colNameT2w=t2wKeyWord
    #                         ,colNameAdc="adc"+spacing_keyword
    #                         ,colNameHbv="hbv"+spacing_keyword
    #                         ,sizeWord=sizeWord
    #                         ,targetSize=targetSize
    #                         ,ToBedivisibleBy32=False
    #                         ,labelColName="label"+spacing_keyword
    #                         )  ,list(df.iterrows())) )

    # pathsLabels = list(map(lambda tupl: tupl[0], resList ))
    # pathst2w = list(map(lambda tupl: tupl[1], resList ))
    # pathsadc = list(map(lambda tupl: tupl[2], resList ))
    # pathshbv = list(map(lambda tupl: tupl[3], resList ))
    # pathsJoined = list(map(lambda tupl: tupl[4], resList ))





    label_name=f"label_{spacing_keyword}" 

    t2wColName="t2w"+spacing_keyword+"cropped"
    adcColName="adc"+spacing_keyword+"cropped"
    hbvColName="hbv"+spacing_keyword+"cropped"
    joinedColName="joined"+spacing_keyword+"cropped"
    print("*********************")

    df[label_name]=   df.apply(partial(resize_and_join
                                ,colNameT2w=t2wKeyWord
                                ,colNameAdc="adc"+spacing_keyword
                                ,colNameHbv="hbv"+spacing_keyword
                                ,sizeWord=""
                                ,targetSize=targetSize
                                ,ToBedivisibleBy32=False
                                ,labelColName=labelColName
                                ,tuplNum=0
                                ), axis=1).compute()
    df[t2wColName]=   df.apply(partial(resize_and_join
                                ,colNameT2w=t2wKeyWord
                                ,colNameAdc="adc"+spacing_keyword
                                ,colNameHbv="hbv"+spacing_keyword
                                ,sizeWord=""
                                ,targetSize=targetSize
                                ,ToBedivisibleBy32=False
                                ,labelColName=labelColName
                                ,tuplNum=1
                                ), axis=1).compute()
    df[adcColName]=   df.apply(partial(resize_and_join
                                ,colNameT2w=t2wKeyWord
                                ,colNameAdc="adc"+spacing_keyword
                                ,colNameHbv="hbv"+spacing_keyword
                                ,sizeWord=""
                                ,targetSize=targetSize
                                ,ToBedivisibleBy32=False
                                ,labelColName=labelColName
                                ,tuplNum=2
                                ), axis=1).compute()
    df[hbvColName]=   df.apply(partial(resize_and_join
                                ,colNameT2w=t2wKeyWord
                                ,colNameAdc="adc"+spacing_keyword
                                ,colNameHbv="hbv"+spacing_keyword
                                ,sizeWord=""
                                ,targetSize=targetSize
                                ,ToBedivisibleBy32=False
                                ,labelColName=labelColName
                                ,tuplNum=3
                                ), axis=1).compute()
    #adding dummy labels                            
    dummyLabelPath,img_size=getDummy(df,spacing_keyword)
  
    df[label_name+'fi']=df.apply(partial(addDummyLabelPath,labelName=label_name ,dummyLabelPath= dummyLabelPath ), axis=1).compute()

    # df[joinedColName]=   df.apply(partial(resize_and_join
    #                             ,colNameT2w=t2wKeyWord
    #                             ,colNameAdc="adc"+spacing_keyword
    #                             ,colNameHbv="hbv"+spacing_keyword
    #                             ,sizeWord=""
    #                             ,targetSize=targetSize
    #                             ,ToBedivisibleBy32=False
    #                             ,labelColName=labelColName
    #                             ,tuplNum=4
    #                             ), axis=1).compute()




    # df[label_name]=   df.apply(lambda row : row['multiPaths'][0], axis=1).compute()
    # df[t2wColName]=   df.apply(lambda row : row['multiPaths'][1], axis=1).compute()
    # df[adcColName]=   df.apply(lambda row : row['multiPaths'][2], axis=1).compute()
    # df[hbvColName]=   df.apply(lambda row : row['multiPaths'][3], axis=1).compute()
    # df[joinedColName]=   df.apply(lambda row : row['multiPaths'][4], axis=1).compute()


    # df[[label_name, t2wColName, adcColName,hbvColName,joinedColName  ]] = df['multiPaths'].apply(pd.Series)

    # df[label_name]=pathsLabels
    # df[t2wColName]=pathst2w
    # df[adcColName]=pathsadc
    # df[hbvColName]=pathshbv
    # df[joinedColName]=pathsJoined


    # Standardize.iterateAndpadLabels(df,"label"+spacing_keyword,targetSize, 0.0,spacing_keyword+sizeWord+'_3Chan',False)






# some preprocessing common for all



# bias field correction
# Standardize.iterateAndBiasCorrect('t2w',df)
# Standarization

for keyWord in ['t2w','adc', 'hbv']: #'cor',,'sag'
    # denoising
    # Standardize.iterateAndDenoise(keyWord,df)
    # standarization
    Standardize.iterateAndStandardize(keyWord,df,trainedModelsBasicPath,300)   
#getting dask frame
df = dd.from_pandas(df, npartitions=os.cpu_count())


# standardize labels
Standardize.iterateAndchangeLabelToOnes(df)

#### 
#now registration of adc and hbv to t2w
for keyWord in ['adc','hbv']:
    df['registered_'+keyWord] = df.apply(partial(reg_adc_hbv_to_t2w,colName=keyWord,elacticPath=elacticPath,reg_prop=reg_prop,t2wColName='t2w'), axis=1).compute()

    # resList=[]     
    # with mp.Pool(processes = mp.cpu_count()) as pool:
    #     resList=pool.map(partial(reg_adc_hbv_to_t2w,colName=keyWord,elacticPath=elacticPath,reg_prop=reg_prop,t2wColName='t2w'),list(df.iterrows()))    

    # df['registered_'+keyWord]=resList      
    # pathss = list(map(lambda tupl :tupl[0],resList   ))
    # reg_values = list(map(lambda tupl :tupl[1],resList   ))

    # df['registered_'+keyWord]=pathss  
    # df['registered_'+keyWord+"score"]=reg_values  
#adding data about number of lesions that algorithm should detect
df=semisuperPreprosess.iterate_and_addLesionNumber(df)
df['num_lesions_to_retain_bin']=df.apply(lambda el: np.int(el['case_csPCa']=='YES'), axis=1 ).compute()#binarizing the output

#checking registration by reading from logs the metrics so we will get idea how well it went

#######      
# targetSpacinggg=(spacingDict['t2w_spac_x'][3],spacingDict['t2w_spac_y'][3],spacingDict['t2w_spac_z'][3])
#preprocess_diffrent_spacings(df,targetSpacinggg,"_med_spac_b")

preprocess_diffrent_spacings(df,(0.5,0.5,0.5),"_half_spac_c")
preprocess_diffrent_spacings(df,(1.0,1.0,1.0),"_one_spac_c")
preprocess_diffrent_spacings(df,(1.5,1.5,1.5),"_one_and_half_spac_c")
preprocess_diffrent_spacings(df,(2.0,2.0,2.0),"_two_spac_c")



print("fiiiniiished")
filePath='/home/sliceruser/data/metadata/processedMetaData_current_b.csv'
if os.path.exists(filePath):
    os.remove(filePath) 
df.compute().to_csv(filePath)#.compute()



# /home/sliceruser/data/orig/10032/10032_1000032_adc_med_spac_for_adc_med_spac/result.0.mha
# /home/sliceruser/data/orig/10032/10032_1000032_hbv_med_spac_for_hbv_med_spac/result.0.mha
    
# orig/10012/10012_1000012_adc_med_spac_for_adc_med_spac/result.0.mha
# orig/10012/10012_1000012_hbv_med_spac_for_hbv_med_spac/result.0.mha







