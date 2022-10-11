from pathlib import Path

import numpy as np
import SimpleITK as sitk

#from https://github.com/DIAGNijmegen/picai_eval/blob/140b86452283ecd049e2138a738ba224fda9b936/src/picai_eval/image_utils.py#L77
def read_image(path):
    """Read image, given a filepath"""
    # return sitk.GetArrayFromImage(sitk.ReadImage(path))
    return np.load(path)

def read_prediction(path):
    """Read prediction, given a filepath"""
    # read prediction and ensure correct dtype
    return np.array(read_image(path), dtype=np.float32)


def read_label(path) :
    """Read label, given a filepath"""
    # read label and ensure correct dtype
    return np.array(read_image(path), dtype=np.int32)
