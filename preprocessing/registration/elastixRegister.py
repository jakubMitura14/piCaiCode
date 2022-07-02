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

def reg_adc_hbv_to_t2w(row,colName,elacticPath,reg_prop,t2wColName ):
    """
    registers adc and hbv images to t2w image
    first we need to create directories for the results
    then we suply hbv and adc as moving images and t2w as static one - and register in reference to it
    we do it in multiple threads at once and we waiteach time the process finished
    """
    row=row[1]
    patId=str(row['patient_id'])
    path=str(row[colName])
    outPath = path.replace(".mha","_for_"+colName)
    result=pathOs.join(outPath,"result0.mha")
    #returning faster if the result is already present
    if(pathOs.exists(result)):
        print("registered already present")
        return result     

    if(len(path)>1):
        
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

