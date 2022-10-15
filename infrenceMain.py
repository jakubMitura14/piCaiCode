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

from evalutils import SegmentationAlgorithm
from evalutils.validators import (UniqueImagesValidator,
                                  UniquePathIndicesValidator)
import Three_chan_baseline_hyperParam
from pathlib import Path    

checkPointPath="/path/to/checkpoint.ckpt"
normalizationsDir="" #path to the files with normalization data
elacticPath='/home/sliceruser/elastixBase/elastix-5.0.1-linux/bin/elastix'
reg_prop=""
physical_size =(81.0, 160.0, 192.0)#taken from picai used to crop image so only center will remain
options = Three_chan_baseline_hyperParam.getOptions()
spacing_keyword=options["spacing_keyword"][0]
model=options["models"][0]
regression_channels=options["regression_channels"][0]
tempPath=""

"""
accepts row from dataframe where paths to the files are present 
"""
def standardize(row):
    for seriesString in ['t2w','adc', 'hbv']: 
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





def infrence():
    
    model = model.LigtningModel.Model.load_from_checkpoint("/path/to/checkpoint.ckpt")
    # disable randomness, dropout, etc...
    model.eval()

    # predict with the model
    y_hat = model(x)


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
        self.cspca_detection_map_path = Path("/output/images/cspca-detection-map/cspca_detection_map.mha")
        self.case_confidence_path = Path("/output/cspca-case-level-likelihood.json")

        # input / output paths for nnUNet
        self.nnunet_inp_dir = Path("/opt/algorithm/nnunet/input")
        self.nnunet_out_dir = Path("/opt/algorithm/nnunet/output")
        self.nnunet_results = Path("/opt/algorithm/results")

        # ensure required folders exist
        self.nnunet_inp_dir.mkdir(exist_ok=True, parents=True)
        self.nnunet_out_dir.mkdir(exist_ok=True, parents=True)
        self.cspca_detection_map_path.parent.mkdir(exist_ok=True, parents=True)

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

    def preprocess_input(self):
        """Preprocess input images to nnUNet Raw Data Archive format"""
        # set up Sample
        sample = Sample(
            scans=[
                sitk.ReadImage(str(path))
                for path in self.scan_paths
            ],
        )

        # perform preprocessing
        sample.preprocess()

        # write preprocessed scans to nnUNet input directory
        for i, scan in enumerate(sample.scans):
            path = self.nnunet_inp_dir / f"scan_{i:04d}.nii.gz"
            atomic_image_write(scan, path)

    # Note: need to overwrite process because of flexible inputs, which requires custom data loading
    def process(self):
        """
        Load bpMRI scans and generate detection map for clinically significant prostate cancer
        """
        # perform preprocessing
        self.preprocess_input()

        # perform inference using nnUNet
        pred_ensemble = None
        ensemble_count = 0
        for trainer in [
            "nnUNetTrainerV2_Loss_FL_and_CE_checkpoints",
        ]:
            # predict sample
            self.predict(
                task="Task2203_picai_baseline",
                trainer=trainer,
                checkpoint="model_best",
            )

            # read softmax prediction
            pred_path = str(self.nnunet_out_dir / "scan.npz")
            pred = np.array(np.load(pred_path)['softmax'][1]).astype('float32')
            os.remove(pred_path)
            if pred_ensemble is None:
                pred_ensemble = pred
            else:
                pred_ensemble += pred
            ensemble_count += 1

        # average the accumulated confidence scores
        pred_ensemble /= ensemble_count

        # the prediction is currently at the size and location of the nnU-Net preprocessed
        # scan, so we need to convert it to the original extent before we continue
        convert_to_original_extent(
            pred=pred_ensemble,
            pkl_path=self.nnunet_out_dir / "scan.pkl",
            dst_path=self.nnunet_out_dir / "softmax.nii.gz",
        )

        # now each voxel in softmax.nii.gz corresponds to the same voxel in the original (T2-weighted) scan
        pred_ensemble = sitk.ReadImage(str(self.nnunet_out_dir / "softmax.nii.gz"))

        # extract lesion candidates from softmax prediction
        # note: we set predictions outside the central 81 x 192 x 192 mm to zero, as this is far outside the prostate
        detection_map = extract_lesion_candidates_cropped(
            pred=sitk.GetArrayFromImage(pred_ensemble),
            threshold="dynamic"
        )

        # convert detection map to a SimpleITK image and infuse the physical metadata of original T2-weighted scan
        reference_scan_original_path = str(self.scan_paths[0])
        reference_scan_original = sitk.ReadImage(reference_scan_original_path)
        detection_map: sitk.Image = sitk.GetImageFromArray(detection_map)
        detection_map.CopyInformation(reference_scan_original)

        # save prediction to output folder
        atomic_image_write(detection_map, str(self.cspca_detection_map_path))

        # save case-level likelihood
        with open(self.case_confidence_path, 'w') as fp:
            json.dump(float(np.max(sitk.GetArrayFromImage(detection_map))), fp)

    def predict(self, task, trainer="nnUNetTrainerV2", network="3d_fullres",
                checkpoint="model_final_checkpoint", folds="0,1,2,3,4", store_probability_maps=True,
                disable_augmentation=False, disable_patch_overlap=False):
        """
        Use trained nnUNet network to generate segmentation masks
        """

        # Set environment variables
        os.environ['RESULTS_FOLDER'] = str(self.nnunet_results)

        # Run prediction script
        cmd = [
            'nnUNet_predict',
            '-t', task,
            '-i', str(self.nnunet_inp_dir),
            '-o', str(self.nnunet_out_dir),
            '-m', network,
            '-tr', trainer,
            '--num_threads_preprocessing', '2',
            '--num_threads_nifti_save', '1'
        ]

        if folds:
            cmd.append('-f')
            cmd.extend(folds.split(','))

        if checkpoint:
            cmd.append('-chk')
            cmd.append(checkpoint)

        if store_probability_maps:
            cmd.append('--save_npz')

        if disable_augmentation:
            cmd.append('--disable_tta')

        if disable_patch_overlap:
            cmd.extend(['--step_size', '1'])

        subprocess.check_call(cmd)


if __name__ == "__main__":
    csPCaAlgorithm().process()