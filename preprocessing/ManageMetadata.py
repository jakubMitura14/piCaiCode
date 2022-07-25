import functools
import importlib.util
import multiprocessing as mp
import os
import sys
from functools import partial
import numpy as np
import pandas as pd
import SimpleITK as sitk


def loadLib(name,path):
    spec = importlib.util.spec_from_file_location(name, path)
    res = importlib.util.module_from_spec(spec)
    sys.modules[name] = res
    spec.loader.exec_module(res)
    return res
    
#transformsForMain =loadLib("transformsForMain", "/home/sliceruser/data/piCaiCode/preprocessing/transformsForMain.py")


def getMonaiSubjectDataFromDataFrame(row,chan3_col_name,label_name,chan3_col_name_val,label_name_val):
        """
        given row from data frame prepares Subject object from it
        """
        subject= {"chan3_col_name": str(row[chan3_col_name])
        ,"chan3_col_name_val": str(row[chan3_col_name_val])        
        #, "cor":str(row['cor'])
        #, "hbv":str(row[hbv_name])
        #, "sag":str(row['sag'])
        #, "t2w":str(row[t2w_name])
        # , "isAnythingInAnnotated":row['isAnythingInAnnotated']
        , "patient_id":str(row['patient_id'])
        # , "study_id":row['study_id']
        # , "patient_age":row['patient_age']
        # , "psa":row['psa']
        # , "psad":row['psad']
        # , "prostate_volume":row['prostate_volume']
        # , "histopath_type":row['histopath_type']
        # , "lesion_GS":row['lesion_GS']
        , "label":str(row[label_name])
        , "label_name_val":str(row[label_name_val])
        
        
        }

        return subject


def load_df_only_full(df,chan3_col_name,label_name,is_whole_to_train):
    df = df.loc[df['isAnyMissing'] ==False]
    df = df.loc[df['isAnythingInAnnotated']>0 ]
    # deficientPatIDs=[]
    # data_dicts = list(map(lambda row: getMonaiSubjectDataFromDataFrame(row[1],chan3_col_name,label_name)  , list(df.iterrows())))
    # train_transforms=transformsForMain.get_train_transforms(0.1#RandGaussianNoised_prob
    #                                                         ,0.1#RandAdjustContrastd_prob
    #                                                         ,0.1#RandGaussianSmoothd_prob
    #                                                         ,0.1#RandRicianNoised_prob
    #                                                         ,0.1#RandFlipd_prob
    #                                                         ,0.1#RandAffined_prob
    #                                                         ,0.1#RandCoarseDropoutd_prob
    #                                                         ,is_whole_to_train )
    # val_transforms= transformsForMain.get_val_transforms(is_whole_to_train)

    # for dictt in data_dicts:    
    #     try:
    #         dat = train_transforms(dictt)
    #         dat = val_transforms(dictt)
    #     except:
    #         print("error loading image")
    #         dat = train_transforms(dictt)# TODO remove
    #         dat = val_transforms(dictt)# TODO remove            
    #         pass
    #         #deficientPatIDs.append(dictt['patient_id'])
    #         #print(dictt['patient_id'])


    # def isInDeficienList(row):
    #         return row['patient_id'] not in deficientPatIDs

    # df["areTransformsNotDeficient"]= df.apply(lambda row : isInDeficienList(row), axis = 1)  

    # df = df.loc[ df['areTransformsNotDeficient']]

    return df



def get_size_meta(row,colName):
    row=row[1]
    patId=str(row['patient_id'])
    path=str(row[colName])
    if(len(path)>1):
        image = sitk.ReadImage(path)
        sizz= image.GetSize()
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
