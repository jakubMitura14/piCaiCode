import pandas as pd
import SimpleITK as sitk
from KevinSR import mask_interpolation, SOUP_GAN
import os
import numpy as np
from scipy import ndimage, interpolate
from scipy.ndimage import zoom
from KevinSR import SOUP_GAN
import monai
import SimpleITK as sitk
from copy import deepcopy
import tensorflow as tf
from numba import cuda 

def copyDirAndOrigin(imageOrig,spacing,data):
    image1 = sitk.GetImageFromArray(data)
    image1.SetSpacing(spacing) #updating spacing
    image1.SetOrigin(imageOrig.GetOrigin())
    image1.SetDirection(imageOrig.GetDirection()) 
    #print(image1.GetSize())
    return image1




#pathT2w,pathHbv,pathADC,patht2wLabel
def testResample(path, targetSpac):
    imageOrig = sitk.ReadImage(path)
    origSize= imageOrig.GetSize()
    orig_spacing=imageOrig.GetSpacing()
    currentSpacing = list(orig_spacing)
    print(f"origSize {origSize}")
    #new size of the image after changed spacing
    new_size = tuple([int(origSize[0]*(orig_spacing[0]/targetSpac[0])),
                    int(origSize[1]*(orig_spacing[1]/targetSpac[1])),
                    int(origSize[2]*(orig_spacing[2]/targetSpac[2]) )  ]  )
    print(f"new_size {new_size}")
    anySuperSampled = False
    data=sitk.GetArrayFromImage(imageOrig)
    #supersampling if needed
    for axis in [0,1,2]:    
        if(new_size[axis]>origSize[axis]):
            anySuperSampled=True
            #in some cases the GPU memory is not cleared enough
            device = cuda.get_current_device()
            device.reset()
            currentSpacing[axis]=targetSpac[axis]
            pre_slices = origSize[axis]
            post_slices = new_size[axis]
            Z_FAC = post_slices/pre_slices # Sampling factor in Z direction
            if(axis==1):
                data = np.moveaxis(data, 1, 2)
            if(axis==2):
                data = np.moveaxis(data, 0, 2)
            #Call the SR interpolation tool from KevinSR
            print(f"thicks_ori shape {data.shape} ")

            data = SOUP_GAN(data, Z_FAC,1)
            print(f"thins_gen shape {data.shape} ")
            if(axis==1):
                data = np.moveaxis(data, 2, 1)
            if(axis==2):
                data = np.moveaxis(data, 2, 0)            
            

    #we need to recreate itk image object only if some supersampling was performed
    if(anySuperSampled):
        image=copyDirAndOrigin(imageOrig,tuple(currentSpacing),data)
    else:
        image=imageOrig
    #copmpleting resampling given some subsampling needs to be performed
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(targetSpac)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetSize(new_size)
    return resample.Execute(image)
    
    
