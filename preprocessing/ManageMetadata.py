import pandas as pd
import SimpleITK as sitk
import numpy as np
import transformsForMain# as transformsForMain
# from transformsForMain import get_train_transforms
# from transformsForMain import get_val_transforms
import os
import multiprocessing as mp
import functools
from functools import partial

def getMonaiSubjectDataFromDataFrame(row,t2w_name,adc_name,hbv_name,label_name):
        """
        given row from data frame prepares Subject object from it
        """
        subject= {"adc": str(row[adc_name])        
        #, "cor":str(row['cor'])
        , "hbv":str(row[hbv_name])
        #, "sag":str(row['sag'])
        , "t2w":str(row[t2w_name])
        # , "isAnythingInAnnotated":row['isAnythingInAnnotated']
        # , "patient_id":row['patient_id']
        # , "study_id":row['study_id']
        # , "patient_age":row['patient_age']
        # , "psa":row['psa']
        # , "psad":row['psad']
        # , "prostate_volume":row['prostate_volume']
        # , "histopath_type":row['histopath_type']
        # , "lesion_GS":row['lesion_GS']
        , "label":str(row[label_name])
        
        
        }

        return subject


def load_df_only_full(df,t2w_name,adc_name,hbv_name,label_name):
    df = df.loc[df['isAnyMissing'] ==False]
    df = df.loc[df['isAnythingInAnnotated']>0 ]
    deficientPatIDs=[]
    data_dicts = list(map(lambda row: getMonaiSubjectDataFromDataFrame(row[1],t2w_name,adc_name,hbv_name,label_name)  , list(df.iterrows())))
    train_transforms=transformsForMain.get_train_transforms()
    val_transforms= transformsForMain.get_val_transforms()

    for dictt in data_dicts:    
        try:
            dat = train_transforms(dictt)
            dat = val_transforms(dictt)
        except:
            deficientPatIDs.append(dictt['patient_id'])
            print(dictt['patient_id'])


    def isInDeficienList(row):
            return row['patient_id'] not in deficientPatIDs

    df["areTransformsNotDeficient"]= df.apply(lambda row : isInDeficienList(row), axis = 1)  

    df = df.loc[ df['areTransformsNotDeficient']]

    return df



def get_size_meta(row,colName):
    row=row[1]
    patId=str(row['patient_id'])
    path=str(row[colName])
    if(len(path)>1):
        image = sitk.ReadImage(path)
        sizz= (image.GetSize(),patId,colName)
        return list(sizz)
    return [-1,-1,-1]
resList=[]

def addSizeMetaDataToDf(keyWord,df):
    resList=[]
    with mp.Pool(processes = mp.cpu_count()) as pool:
        resList=pool.map(partial(get_size_meta,colName=keyWord)  ,list(df.iterrows()))    
    df[keyWord+'_sizz_x']= list(map(lambda arr:arr[0], resList))    
    df[keyWord+'_sizz_y']= list(map(lambda arr:arr[1], resList))    
    df[keyWord+'_sizz_z']= list(map(lambda arr:arr[2], resList))    


def getMaxSize(keyWord,df):
    resList=[]
    max_size_x = np.max(list(filter(lambda it: it>0 ,df[keyWord+'_sizz_x'].to_numpy() )))
    max_size_y = np.max(list(filter(lambda it: it>0 ,df[keyWord+'_sizz_y'].to_numpy() )))
    max_size_z = np.max(list(filter(lambda it: it>0 ,df[keyWord+'_sizz_z'].to_numpy() )))
    return (max_size_x,max_size_y,max_size_z)            