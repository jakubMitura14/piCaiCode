#based on https://github.com/NIH-MIP/Radiology_Image_Preprocessing_for_Deep_Learning/blob/main/Codes/Main_Preprocessing.py
from __future__ import print_function
import argparse
import sys
import json
import glob
import random
from dicom.Dicom_Tools import *
from utils.utils import *
import csv
from utils.Annotation_utils import *
from image.Nyul_preprocessing import *



if args.image_type=='MRI':
            for i, patient in enumerate(Patient_List):
                Input_path = join(args.dicom_folder, patient, 't2')
                if isdir(Input_path):
                        image1 = DicomRead(Input_path)
                        data = sitk.GetArrayFromImage(image1)
                        # shift the data up so that all intensity values turn positive
                        data -= np.min(data)
                        # Removing the outliers with a probability of occuring less than 5e-3 through histogram computation
                        histo, bins = np.histogram(data.flatten(), 10)
                        histo = normalize(histo)
                        Bin = bins[np.min(np.where(histo < 5e-3))]
                        data = np.clip(data, 0, Bin)
                        image = sitk.GetImageFromArray(data)
                        image.SetSpacing(image1.GetSpacing())
                        image.SetOrigin(image1.GetOrigin())
                        image.SetDirection(image1.GetDirection())
                        
                        
                        
                        
                        #Bias Correction
                        
                        def Dicom_Bias_Correct(image):
    """"
    For more information please see: https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html
    """
    maskImage = sitk.OtsuThreshold(image, 0, 1, 200)
    inputImage = sitk.Cast(image, sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # numberFittingLevels = 4
    imageB = corrector.Execute(inputImage, maskImage)
    imageB.SetSpacing(image.GetSpacing())
    imageB.SetOrigin(image.GetOrigin())
    imageB.SetDirection(image.GetDirection())
    return imageB
                        
                        
                        
                        image_B = Dicom_Bias_Correct(image)
                        image_B = image
                        if args.Image_format=='Nifti':
                                NiftiWrite(image_B, join(args.target_folder,'Bias_field_corrected'),output_name = patient+'.nii', OutputPixelType=args.Output_Pixel_Type)
                        elif args.Image_format=='Dicom':
                                DicomWrite(image_B, join(args.target_folder,'Bias_field_corrected',patient),
                                   Referenced_dicom_image_directory=Input_path, OutputPixelType=args.Output_Pixel_Type)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm




def Resample_3D(image,New_Spacing,New_Size=None,OutputPixelType='Uint16',mask=None):
    """
    Image will be resampled to the New_Spacing as the desired size
                    or
    Image will be resampled to the New_Size as the desired size
    :param image:
    :param New_Spacing: is a triple in form of [x,y,z] for spacing in x, y, and z
    :param New_Size: is a triple in form of [x,y,z] for diemsnions in x, y, and z
    :param
    :return:
    """
    if OutputPixelType=='Uint16':
        OutputPixelType=16
    else:
        OutputPixelType=8
    resample = sitk.ResampleImageFilter()
    orig_size = np.array(image.GetSize(), dtype=np.int)
    orig_spacing = image.GetSpacing()
    if len(New_Spacing)==3:
        z=New_Spacing[2]
        x=New_Spacing[0]
        y=New_Spacing[1]
    elif len(New_Spacing)==2:
        z=None
        y=New_Spacing[1]
        x=New_Spacing[0]
    elif len(New_Spacing)==0 and len(New_Size)==3:
        z = orig_size[2] * orig_spacing[2] / New_Size[2]
        x = orig_size[0] * orig_spacing[0] / New_Size[0]
        y = orig_size[1] * orig_spacing[1] / New_Size[1]
    elif len(New_Size)==2:
        z=None
        x = orig_size[0] * orig_spacing[0] / New_Size[0]
        y = orig_size[1] * orig_spacing[1] / New_Size[1]
    else:
        warnings.warn("Warning!!! Potentially wrong arguments for size and spacing... Thus no Resizing")
        z = None
        x=orig_spacing[0]
        y=orig_spacing[1]
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    if z is None:
        z=image.GetSpacing()[-1]
        max_num_slices = orig_size[-1]#
    else:
        max_num_slices = (orig_size[2]*orig_spacing[2]/z).astype(np.int)
    if New_Size is None:
        New_Size=[int(orig_size[0] * orig_spacing[0] /x),int(orig_size[1] * orig_spacing[1] /y)]
    new_size = [New_Size[0], New_Size[1],
                max_num_slices]
    new_size = [int(s) for s in new_size]
    new_spacing=[x, y, z]
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    newimage = resample.Execute(sitk.Cast(image, sitk.sitkFloat32))
    output_type = sitk.sitkInt16
    if OutputPixelType==8:
        output_type = sitk.sitkInt8
    if mask is not None:
        newimagemask =  resample.Execute(sitk.Cast(mask, sitk.sitkFloat32))
        return sitk.Cast(newimage, output_type), sitk.Cast(newimagemask, output_type)
    return sitk.Cast(newimage, output_type)




#nyul
            train(train_patients, dir1=join(args.target_folder,'Bias_field_corrected'),
                               dir2=join(args.target_folder,'trained_model'+args.image_type+'.npz'))
            Model_Path = join(args.target_folder,'trained_model'+args.image_type+'.npz')
            f = np.load(Model_Path, allow_pickle=True)
            Model = f['trainedModel'].all()
            meanLandmarks = Model['meanLandmarks']

            Patient_List = json.load(open(join(args.target_folder,'all_patients.json'),'r'))
            for i, patient in enumerate(Patient_List):

                Input_path = join(args.target_folder, 'Bias_field_corrected', patient)
                print('Standardizing ...', basename(Input_path))
                try:
                    image_B = sitk.ReadImage(Input_path+'.nii')
                except Exception:
                    image_B = DicomRead(Input_path)

                image_B_S= transform(image_B,meanLandmarks=meanLandmarks)