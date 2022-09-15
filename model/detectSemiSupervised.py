"""
idea is to get a small network that will get the output of the main unet 
and get the number reflecting number of present lesions - so we get from segmentation to regression task

so we probably shoul do couple strided convolutions then flatten and linear layer with output 1
in order for linear layer to work we need also to have fixed dimensions - hence we need to calculate them
by reducing the 

"""


import warnings
from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection
from monai.utils import alias, deprecated_arg, export
import functools
import operator
from torch.nn.intrinsic.qat import ConvBnReLU3d

"""
based on example from 
https://datascience.stackexchange.com/questions/40906/determining-size-of-fc-layer-after-conv-layer-in-pytorch

"""
class UNetToRegresion(nn.Module):
    def __init__(self,
        in_channels,
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(ConvBnReLU3d(in_channels=in_channels, out_channels=1, kernel_size=3, stride=2,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            ConvBnReLU3d(in_channels=1, out_channels=1, kernel_size=3, stride=2,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            ConvBnReLU3d(in_channels=1, out_channels=1, kernel_size=3, stride=2,qconfig = torch.quantization.get_default_qconfig('fbgemm')),
            nn.AdaptiveMaxPool3d((8,8,2)),#ensuring such dimension 
            nn.Flatten(),
            nn.BatchNorm3d(8*8*4),
            nn.Linear(in_features=8*8*4, out_features=100),
            nn.BatchNorm3d(100),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=100, out_features=1)
        )
    def forward(self, x):
        return self.model(x)





"""
based on https://github.com/DIAGNijmegen/Report-Guided-Annotation/blob/6909800fbc17bc3d833d91e4977d3baf47975fda/src/report_guided_annotation/create_automatic_annotations.py

"""

# import os
# import json
# import numpy as np
# from tqdm import tqdm
# import concurrent.futures
# from concurrent.futures import ThreadPoolExecutor
# import SimpleITK as sitk
# from typing import Tuple, List, Dict, Union, Optional

# from report_guided_annotation.extract_lesion_candidates import 

# def getPredictedLesions():

#             """
#     Step 1: Extract lesion candidates from softmax prediction
#     Please refer to [1] or [2] for documentation on the lesion candidate extraction.
#     This gives:
#     - confidences: list of tuples with (lesion ID, lesion confidence), e.g. [(1, 0.2321), (2, 0.3453), (3, 0.0431), ...]
#     - indexed_pred: numbered masks for the extracted lesion candidates. Lesion candidates are non-overlapping and the same shape as `pred`.
#                     E.g., for the lesion candidate with confidence of 0.3453, each voxel has the value 2
#     """

#     _, confidences, indexed_pred = extract_lesion_candidates(
#     pred,
#     threshold=threshold,
#     num_lesions_to_extract=num_lesions_to_retain,
# )



# #look here https://github.com/guanhuaw/MIRTorch/blob/master/examples/demo_mnist.ipynb for soft thresholding
