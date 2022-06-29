import SimpleITK as sitk
from subprocess import Popen
import subprocess
import SimpleITK as sitk
import pandas as pd
import multiprocessing as mp
import functools
from functools import partial
import sys

def reg_adc_hbv_to_t2w(row,colName,elacticPath,reg_prop ):
    """
    registers adc and hbv images to t2w image
    first we need to create directories for the results
    then we suply hbv and adc as moving images and t2w as static one - and register in reference to it
    we do it in multiple threads at once and we waiteach time the process finished
    """
    row=row[1]
    patId=str(row['patient_id'])
    path=str(row[colName])
    if(len(path)>1):
        outPath = path.replace(".mha","_for_"+colName)
        cmd='mkdir '+ outPath
        p = Popen(cmd, shell=True)
        p.wait()
        cmd=f"{elacticPath} -f {row['t2w']} -m {path} -out {outPath} -p {reg_prop}"
        print(cmd)
        try:
            p = Popen(cmd, shell=True)
        except:
            print("error in patId")
        p.wait()

