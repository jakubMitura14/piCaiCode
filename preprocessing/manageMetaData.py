import pandas as pd
import preprocessing.transformsForMain# as transformsForMain
# from transformsForMain import get_train_transforms
# from transformsForMain import get_val_transforms


def getMonaiSubjectDataFromDataFrame(row):
        """
        given row from data frame prepares Subject object from it
        """
        subject= {"adc": str(row['adc'])        
        , "cor":str(row['cor'])
        , "hbv":str(row['hbv'])
        , "sag":str(row['sag'])
        , "t2w":str(row['t2w'])
        , "isAnythingInAnnotated":row['isAnythingInAnnotated']
        , "patient_id":row['patient_id']
        , "study_id":row['study_id']
        , "patient_age":row['patient_age']
        , "psa":row['psa']
        , "psad":row['psad']
        , "prostate_volume":row['prostate_volume']
        , "histopath_type":row['histopath_type']
        , "lesion_GS":row['lesion_GS']
        , "label":str(row['reSampledPath'])
        
        
        }

        return subject


def load_df_only_full():
    df = pd.read_csv('/home/sliceruser/labels/processedMetaData.csv')
    df = df.loc[df['isAnyMissing'] ==False]
    df = df.loc[df['isAnythingInAnnotated']>0 ]
    deficientPatIDs=[]
    data_dicts = list(map(lambda row: getMonaiSubjectDataFromDataFrame(row[1])  , list(df.iterrows())))
    train_transforms=preprocessing.transformsForMain.get_train_transforms()
    val_transforms= preprocessing.transformsForMain.get_val_transforms()

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
