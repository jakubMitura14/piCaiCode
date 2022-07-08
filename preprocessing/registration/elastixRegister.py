import SimpleITK as sitk
from subprocess import Popen
import subprocess
import SimpleITK as sitk
import pandas as pd
import multiprocessing as mp
import functools
from functools import partial
import sys
import os.path
from os import path as pathOs
import comet_ml
from comet_ml import Experiment

def reg_adc_hbv_to_t2w(row,colName,elacticPath,reg_prop,t2wColName,experiment=None):
    """
    registers adc and hbv images to t2w image
    first we need to create directories for the results
    then we suply hbv and adc as moving images and t2w as static one - and register in reference to it
    we do it in multiple threads at once and we waiteach time the process finished
    """

    row=row[1]
    study_id=str(row['study_id'])
    
    patId=str(row['patient_id'])
    path=str(row[colName])
    outPath = path.replace(".mha","_for_"+colName)
    result=pathOs.join(outPath,"result.0.mha")
    print("**********  ***********  ****************")
    print(result)

    
    
#     /home/sliceruser/data/orig/10005/10005_1000005_adc_stand_medianSpac_for_adc_med_spac/result.0.mha
    
    
    #     /home/sliceruser/data/orig/10005/10005_1000005_adc_stand_medianSpac_for_adc_med_spac/result.0.mha
#     /home/sliceruser/data/orig/10005/10005_1000005_adc_stand_medianSpac_for_adc_med_spac
#     /home/sliceruser/data/orig/10005/10005_1000005_hbv_stand_medianSpac_for_hbv_med_spac/result.0.mha
    
    print(pathOs.exists(result))
    #returning faster if the result is already present
    #if(pathOs.exists(outPath)):
    if(pathOs.exists(result)):
        if(experiment!=None):
            experiment.log_text(f"already registered {colName} {study_id}")
        
        print("registered already present")
        return result     
    else:
        if(len(path)>1):
            if(experiment!=None):  
                print(f"new register {colName} {study_id}")
                experiment.log_text(f"new register {colName} {study_id}")

            cmd='mkdir '+ outPath
            p = Popen(cmd, shell=True)
            p.wait()
            cmd=f"{elacticPath} -f {row[t2wColName]} -m {path} -out {outPath} -p {reg_prop}"
            print(cmd)
            try:
                p = Popen(cmd, shell=True)
            except:
                print("error in patId")
            p.wait()
            return result
        else:
            return ""    
    return ""
