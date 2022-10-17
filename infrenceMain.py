#https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_basic.html
import model.LigtningModel
from os import listdir
from os import path as pathOs
from os.path import basename, dirname, exists, isdir, join, split
import numpy as np
import pandas as pd
import SimpleITK as sitk
from intensity_normalization.normalize.nyul import NyulNormalize
import dask
import dask.dataframe as dd
import math

from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
import Three_chan_baseline_hyperParam
from pathlib import Path    
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
import numpy as np
import dask
import dask.dataframe as dd


checkPointPath="/path/to/checkpoint.ckpt"
normalizationsDir="" #path to the files with normalization data
elacticPath='/home/sliceruser/elastixBase/elastix-5.0.1-linux/bin/elastix'
reg_prop="" # properties for elastix registration
physical_size =(81.0, 160.0, 192.0)#taken from picai used to crop image so only center will remain
options = Three_chan_baseline_hyperParam.getOptions()
spacing_keyword=options["spacing_keyword"][0]
model=options["models"][0]
regression_channels=options["regression_channels"][0]
tempPath=""


class MissingSequenceError(Exception):
    """Exception raised when a sequence is missing."""

    def __init__(self, name, folder):
        message = f"Could not find scan for {name} in {folder} (files: {os.listdir(folder)})"
        super.__init__(message)


class MultipleScansSameSequencesError(Exception):
    """Exception raised when multiple scans of the same sequences are provided."""

    def __init__(self, name, folder):
        message = f"Found multiple scans for {name} in {folder} (files: {os.listdir(folder)})"
        super.__init__(message)



def copyDirAndOrigin(imageOrig,spacing,data):
    image1 = sitk.GetImageFromArray(data)
    image1.SetSpacing(spacing) #updating spacing
    image1.SetOrigin(imageOrig.GetOrigin())
    image1.SetDirection(imageOrig.GetDirection()) 
    #print(image1.GetSize())
    return image1



"""
accepts row from dataframe where paths to the files are present 
and standardize t2w adc an hbv using saved histograms
"""
def standardize(row,seriesString,tempPath):
    newPath=join(tempPath,Path(row[seriesString]).name)

    pathNormalizer = join(normalizationsDir,seriesString+".npy")
    nyul_normalizer = NyulNormalize()
    nyul_normalizer.load_standard_histogram(pathNormalizer)

    image1=sitk.ReadImage(row[seriesString])
    image1 = sitk.DICOMOrient(image1, 'RAS')
    image1 = sitk.Cast(image1, sitk.sitkFloat32)
    data=nyul_normalizer(sitk.GetArrayFromImage(image1))
    #recreating image keeping relevant metadata
    image = sitk.GetImageFromArray(data)  
    image.SetSpacing(image1.GetSpacing())
    image.SetOrigin(image1.GetOrigin())
    image.SetDirection(image1.GetDirection())

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(newPath)
    writer.Execute(image)    
    return newPath        
"""
register adc and hbv to t2w
keyword is either adc or hbv column names
t2wkeyword column name with t2w path
reIndex - used in recursion
elacticPath - path to elastix
reg_prop - path to elastix properties used in registration
"""
def register(row, keyword,t2wkeyword,elacticPath,reg_prop,reIndex=0):
        
    path=str(row[keyword])
    outPath = path.replace(".mha","_for_"+keyword)
    result=pathOs.join(outPath,"result.0.mha")
    logPath=pathOs.join(outPath,"elastix.log")

    if(len(path)>1):
        #creating the folder if none is present
        if(not pathOs.exists(outPath)):
            cmd='mkdir '+ outPath
            p = Popen(cmd, shell=True)
            p.wait()

        cmd=f"{elacticPath} -f {row[t2wkeyword]} -m {path} -out {outPath} -p {reg_prop} -threads 1"
        print(cmd)
        p = Popen(cmd, shell=True)#,stdout=subprocess.PIPE , stderr=subprocess.PIPE
        p.wait()
        #we will repeat operation multiple max 9 times if the result would not be written
        if((not pathOs.exists(result)) and reIndex<8):
            reIndexNew=reIndex+1
            if(reIndex==4): #in case it do not work we will try diffrent parametrization
                reg_prop=reg_prop.replace("parameters","parametersB")              
            register(row, keyword,t2wkeyword,elacticPath,reg_prop,reIndexNew)
        if(not pathOs.exists(result)):
            print("registration unsuccessfull")
            return " "
        #return path to registered file
        return result #
    #return " " in case of incorrect path input            
    return " "

def resamplBase(path,targetSpacing,interpolator):

    imageOrig = sitk.ReadImage(path)
    origSize= imageOrig.GetSize()
    orig_spacing=imageOrig.GetSpacing()
    currentSpacing = list(orig_spacing)
    print(f"origSize {origSize}")
    #new size of the image after changed spacing
    new_size = tuple([int(origSize[0]*(orig_spacing[0]/targetSpacing[0])),
                    int(origSize[1]*(orig_spacing[1]/targetSpacing[1])),
                    int(origSize[2]*(orig_spacing[2]/targetSpacing[2]) )  ]  )
    print(f"new_size {new_size}  target spacing {targetSpacing}")
    anySuperSampled = False
    data=sitk.GetArrayFromImage(imageOrig)
    image=copyDirAndOrigin(imageOrig,tuple(currentSpacing),data)
    #copmpleting resampling given some subsampling needs to be performed
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(targetSpacing)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(image.GetPixelIDValue())
    resample.SetInterpolator(interpolator)
    resample.SetSize(new_size)
    return resample.Execute(image)

"""
resamples to the given spacing
keyword - column name with file to be resampled
targetSpacing - target spacing
"""
def resample(row,keyword,targetSpacing):
    path=row[keyword]
    spacing_keyword="spac"
    res=resamplBase(path,targetSpacing,sitk.sitkBSpline)
    newPath = path.replace(".mha",spacing_keyword+".mha" )
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(newPath)
    writer.Execute(res)
    return newPath    

"""
resample back to original spacing
"""
def resampleBack(row,keyword,t2wOrig):
    path=row[keyword]
    pathT2w=row[t2wOrig]
    imageT2w= sitk.ReadImage(pathT2w)
    res=resamplBase(path,imageT2w.GetSpacing(),sitk.sitkNearestNeighbor )
    newPath = path.replace(".mha",spacing_keyword+"back.mha" )
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(newPath)
    writer.Execute(res)
    return newPath    



def resizeBase(path,targetSize ):
    paddValue=0
    image1 = sitk.ReadImage(path)              
    currentSize=image1.GetSize()
    sizediffs=(targetSize[0]-currentSize[1]  , targetSize[1]-currentSize[1]  ,targetSize[2]-currentSize[2])
    halfDiffSize=(math.floor(sizediffs[0]/2) , math.floor(sizediffs[1]/2), math.floor(sizediffs[2]/2))
    rest=(sizediffs[0]-halfDiffSize[0]  ,sizediffs[1]-halfDiffSize[1]  ,sizediffs[2]-halfDiffSize[2]  )
    halfDiffSize=np.array(halfDiffSize, dtype='int').tolist() 
    rest=np.array(rest, dtype='int').tolist()    
    #saving only non negative entries
    halfDiffSize_to_pad= list(map(lambda dim : max(dim,0) ,halfDiffSize ))
    rest_to_pad= list(map(lambda dim : max(dim,0) ,rest ))
    #get only negative entries - those mean that we need to crop and we negate it to get positive numbers
    halfDiffSize_to_crop= list(map(lambda dim : (-1)*min(dim,0) ,halfDiffSize ))
    rest_to_crop= list(map(lambda dim : (-1)*min(dim,0) ,rest ))

    padded= sitk.ConstantPad(image1, halfDiffSize_to_pad, rest_to_pad, paddValue)
    return  sitk.Crop(padded, halfDiffSize_to_crop,rest_to_crop )
"""
resize to given physical size taking spacing into account
additionally enable saving the metadata used in this resizing so the process can be reversed in the end
keyword - colname with path of the file to modify
targetSpacing - target spacing
physical_size - physical size irrespective of spacing that the image should have
"""
def resize(row,keyword,targetSize):
    
    
    path=row[keyword]
    res=resizeBase(path,targetSize )


    newPath = path.replace(".mha","cropped.mha" )
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(newPath)
    writer.Execute(res)

    return newPath        

def resizeBack(row,keyword,t2wOrig,resultsDir):
    path=row[keyword]
    pathT2w=row[t2wOrig]
    imageT2w= sitk.ReadImage(pathT2w)
    res=resizeBase(path,imageT2w.GetSize())

    newPath=join(resultsDir,Path(row[t2wOrig]).name)

    newPath = path.replace(".mha","cropped_back.mha" )
    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()
    writer.SetFileName(newPath)
    writer.Execute(res)

    return newPath            


"""
applying pytorch lightning model using checkpoint from path
t2wKey,adcKey,hbvKey - column names with paths to respective modalities
"""
def applyModel(row, t2wKey,adcKey,hbvKey):
    
    path=row[t2wKey]
    image1 = sitk.ReadImage(path)   
    dat=sitk.GetArrayFromImage(image1)
    data= np. zeros_like(dat)#for debugging just return zeros of proper size
    image = sitk.GetImageFromArray(data)  
    image.SetSpacing(image1.GetSpacing())
    image.SetOrigin(image1.GetOrigin())
    image.SetDirection(image1.GetDirection())

    newPath = path.replace(".mha","result.mha" )
    writer = sitk.ImageFileWriter()
    writer.SetFileName(newPath)
    writer.Execute(image)    
    return newPath
    # model = model.LigtningModel.Model.load_from_checkpoint("/path/to/checkpoint.ckpt")
    # # disable randomness, dropout, etc...
    # model.eval()

    # # predict with the model
    # y_hat = model(x)


"""
combines peprocessing infrence and postprocessing
"""
def process(targetSpacing,df,physical_size ,elacticPath,reg_prop,resultsDir):
    sizzX= physical_size[0]/targetSpacing[0]
    sizzY= physical_size[1]/targetSpacing[1]
    sizzZ= physical_size[2]/targetSpacing[2]
    sizz=(sizzX,sizzY,sizzZ) 

    multNum=32#32
    targetSize=(math.ceil(sizz[0]/multNum)*multNum, math.ceil(sizz[1]/multNum)*multNum,math.ceil(sizz[2]/multNum)*multNum  )
    for seriesString in ['t2w','adc', 'hbv']: 
        df[seriesString+"_Stand"]=df.apply(partial(standardize, seriesString= seriesString,tempPath=tempPath )  ,axis=1)
    for keyword in ['adc_Stand', 'hbv_Stand']: 
        df[keyword+"_reg"]=df.apply(partial(register, keyword=keyword,t2wkeyword='t2w_Stand',elacticPath=elacticPath,reg_prop=reg_prop)  ,axis=1)
    for keyword in ['t2w_Stand','adc_Stand_reg', 'hbv_Stand_reg']: 
        df[keyword+"_resampl"]=df.apply(partial(resample, keyword=keyword,targetSpacing=targetSpacing)  ,axis=1)
    for keyword in ['t2w_Stand_resampl','adc_Stand_reg_resampl', 'hbv_Stand_reg_resampl']: 
        df[keyword+"_resiz"]=df.apply(partial(resize, keyword=keyword,targetSize=targetSize)  ,axis=1)
    df['res']=df.apply(partial(applyModel, t2wKey='t2w_Stand_resampl_resiz',adcKey='adc_Stand_reg_resampl_resiz',hbvKey='hbv_Stand_reg_resampl_resiz')  ,axis=1)
    df['res_resampl']=df.apply(partial(resampleBack, keyword='res', t2wOrig='t2w')  ,axis=1)
    df['res_resampl_resize']=df.apply(partial(resizeBack, keyword='res_resampl', t2wOrig='t2w',resultsDir=resultsDir)  ,axis=1)


        

class csPCaAlgorithm(SegmentationAlgorithm):
    """
    Wrapper to deploy trained baseline nnU-Net model from
    https://github.com/DIAGNijmegen/picai_baseline as a
    grand-challenge.org algorithm.
    """

    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )

        # input / output paths for algorithm
        self.image_input_dirs = [
            "/input/images/transverse-t2-prostate-mri",
            "/input/images/transverse-adc-prostate-mri",
            "/input/images/transverse-hbv-prostate-mri",
        ]
        self.scan_paths = []
        # self.cspca_detection_map_path = Path("/output/images/cspca-detection-map/cspca_detection_map.mha")
        # self.case_confidence_path = Path("/output/cspca-case-level-likelihood.json")

        # # input / output paths for nnUNet
        # self.nnunet_inp_dir = Path("/opt/algorithm/nnunet/input")
        # self.nnunet_out_dir = Path("/opt/algorithm/nnunet/output")
        # self.nnunet_results = Path("/opt/algorithm/results")

        # # ensure required folders exist
        # self.nnunet_inp_dir.mkdir(exist_ok=True, parents=True)
        # self.nnunet_out_dir.mkdir(exist_ok=True, parents=True)
        # self.cspca_detection_map_path.parent.mkdir(exist_ok=True, parents=True)

        # input validation for multiple inputs
        scan_glob_format = "*.mha"
        for folder in self.image_input_dirs:
            file_paths = list(Path(folder).glob(scan_glob_format))
            if len(file_paths) == 0:
                raise MissingSequenceError(name=folder.split("/")[-1], folder=folder)
            elif len(file_paths) >= 2:
                raise MultipleScansSameSequencesError(name=folder.split("/")[-1], folder=folder)
            else:
                # append scan path to algorithm input paths
                self.scan_paths += [file_paths[0]]
        print(self.scan_paths)


    # Note: need to overwrite process because of flexible inputs, which requires custom data loading
    def process(self):
        """
        Load bpMRI scans and generate detection map for clinically significant prostate cancer
        """
        print("processss")
        # # perform preprocessing
        # self.preprocess_input()

        # # perform inference using nnUNet
        # pred_ensemble = None
        # ensemble_count = 0
        # for trainer in [
        #     "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
        # ]:
        #     # predict sample
        #     self.predict(
        #         task="Task2203_picai_baseline",
        #         trainer=trainer,
        #         checkpoint="model_best",
        #     )

        #     # read softmax prediction
        #     pred_path = str(self.nnunet_out_dir / "scan.npz")
        #     pred = np.array(np.load(pred_path)['softmax'][1]).astype('float32')
        #     os.remove(pred_path)
        #     if pred_ensemble is None:
        #         pred_ensemble = pred
        #     else:
        #         pred_ensemble += pred
        #     ensemble_count += 1

        # # average the accumulated confidence scores
        # pred_ensemble /= ensemble_count

        # # the prediction is currently at the size and location of the nnU-Net preprocessed
        # # scan, so we need to convert it to the original extent before we continue
        # convert_to_original_extent(
        #     pred=pred_ensemble,
        #     pkl_path=self.nnunet_out_dir / "scan.pkl",
        #     dst_path=self.nnunet_out_dir / "softmax.nii.gz",
        # )

        # # now each voxel in softmax.nii.gz corresponds to the same voxel in the original (T2-weighted) scan
        # pred_ensemble = sitk.ReadImage(str(self.nnunet_out_dir / "softmax.nii.gz"))

        # # extract lesion candidates from softmax prediction
        # # note: we set predictions outside the central 81 x 192 x 192 mm to zero, as this is far outside the prostate
        # detection_map = extract_lesion_candidates_cropped(
        #     pred=sitk.GetArrayFromImage(pred_ensemble),
        #     threshold="dynamic"
        # )

        # # convert detection map to a SimpleITK image and infuse the physical metadata of original T2-weighted scan
        # reference_scan_original_path = str(self.scan_paths[0])
        # reference_scan_original = sitk.ReadImage(reference_scan_original_path)
        # detection_map: sitk.Image = sitk.GetImageFromArray(detection_map)
        # detection_map.CopyInformation(reference_scan_original)

        # # save prediction to output folder
        # atomic_image_write(detection_map, str(self.cspca_detection_map_path))

        # # save case-level likelihood
        # with open(self.case_confidence_path, 'w') as fp:
        #     json.dump(float(np.max(sitk.GetArrayFromImage(detection_map))), fp)

    # def predict(self, task, trainer="nnUNetTrainerV2", network="3d_fullres",
    #             checkpoint="model_final_checkpoint", folds="0,1,2,3,4", store_probability_maps=True,
    #             disable_augmentation=False, disable_patch_overlap=False):
    #     """
    #     Use trained nnUNet network to generate segmentation masks
    #     """

    #     # Set environment variables
    #     os.environ['RESULTS_FOLDER'] = str(self.nnunet_results)

    #     # Run prediction script
    #     cmd = [
    #         'nnUNet_predict',
    #         '-t', task,
    #         '-i', str(self.nnunet_inp_dir),
    #         '-o', str(self.nnunet_out_dir),
    #         '-m', network,
    #         '-tr', trainer,
    #         '--num_threads_preprocessing', '2',
    #         '--num_threads_nifti_save', '1'
    #     ]

    #     if folds:
    #         cmd.append('-f')
    #         cmd.extend(folds.split(','))

    #     if checkpoint:
    #         cmd.append('-chk')
    #         cmd.append(checkpoint)

    #     if store_probability_maps:
    #         cmd.append('--save_npz')

    #     if disable_augmentation:
    #         cmd.append('--disable_tta')

    #     if disable_patch_overlap:
    #         cmd.extend(['--step_size', '1'])

    #     subprocess.check_call(cmd)


if __name__ == "__main__":
    csPCaAlgorithm().process()