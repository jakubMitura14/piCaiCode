"""
based on https://github.com/DIAGNijmegen/Report-Guided-Annotation/blob/6909800fbc17bc3d833d91e4977d3baf47975fda/src/report_guided_annotation/create_automatic_annotations.py

"""

import os
import json
import numpy as np
from tqdm import tqdm
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import SimpleITK as sitk
from typing import Tuple, List, Dict, Union, Optional

from report_guided_annotation.extract_lesion_candidates import 

def getPredictedLesions():

            """
    Step 1: Extract lesion candidates from softmax prediction
    Please refer to [1] or [2] for documentation on the lesion candidate extraction.
    This gives:
    - confidences: list of tuples with (lesion ID, lesion confidence), e.g. [(1, 0.2321), (2, 0.3453), (3, 0.0431), ...]
    - indexed_pred: numbered masks for the extracted lesion candidates. Lesion candidates are non-overlapping and the same shape as `pred`.
                    E.g., for the lesion candidate with confidence of 0.3453, each voxel has the value 2
    """

    _, confidences, indexed_pred = extract_lesion_candidates(
    pred,
    threshold=threshold,
    num_lesions_to_extract=num_lesions_to_retain,
)



#look here https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_mnist.ipynb for soft thresholding
