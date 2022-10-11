from pathlib import Path

import numpy as np
import SimpleITK as sitk

import concurrent.futures
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (Callable, Dict, Hashable, Iterable, List, Optional, Sized,
                    Tuple, Union)
from picai_eval.analysis_utils import (calculate_dsc, calculate_iou,
                                       label_structure, parse_detection_map)
from picai_eval.image_utils import (read_label, read_prediction,
                                    resize_image_with_crop_or_pad)
from picai_eval.metrics import Metrics

from picai_eval.eval import evaluate_case
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




def evaluate_all_cases(listPerEval):
    case_target: Dict[Hashable, int] = {}
    case_weight: Dict[Hashable, float] = {}
    case_pred: Dict[Hashable, float] = {}
    lesion_results: Dict[Hashable, List[Tuple[int, float, float]]] = {}
    lesion_weight: Dict[Hashable, List[float]] = {}

    meanPiecaiMetr_auroc=0.0
    meanPiecaiMetr_AP=0.0
    meanPiecaiMetr_score=0.0

    idx=0
    if(len(listPerEval)>0):
        for pairr in listPerEval:
            idx+=1
            lesion_results_case, case_confidence = pairr

            case_weight[idx] = 1.0
            case_pred[idx] = case_confidence
            if len(lesion_results_case):
                case_target[idx] = np.max([a[0] for a in lesion_results_case])
            else:
                case_target[idx] = 0

            # accumulate outputs
            lesion_results[idx] = lesion_results_case
            lesion_weight[idx] = [1.0] * len(lesion_results_case)

        # collect results in a Metrics object
        valid_metrics = Metrics(
            lesion_results=lesion_results,
            case_target=case_target,
            case_pred=case_pred,
            case_weight=case_weight,
            lesion_weight=lesion_weight
        )
        # for i in range(0,numIters):
        #     valid_metrics = evaluate(y_det=self.list_yHat_val[i*numPerIter:min((i+1)*numPerIter,lenn)],
        #                         y_true=self.list_gold_val[i*numPerIter:min((i+1)*numPerIter,lenn)],
        #                         num_parallel_calls= min(numPerIter,os.cpu_count())
        #                         ,verbose=1
        #                         #,y_true_postprocess_func=lambda pred: pred[1,:,:,:]
        #                         #y_true=iter(y_true),
        #                         ,y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
        #                         #,y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0]
        #                         )
        # meanPiecaiMetr_auroc_list.append(valid_metrics.auroc)
        # meanPiecaiMetr_AP_list.append(valid_metrics.AP)
        # meanPiecaiMetr_score_list.append((-1)*valid_metrics.score)
        #print("finished evaluating")

        meanPiecaiMetr_auroc=valid_metrics.auroc
        meanPiecaiMetr_AP=valid_metrics.AP
        meanPiecaiMetr_score=(-1)*valid_metrics.score
    return (meanPiecaiMetr_auroc,meanPiecaiMetr_AP,meanPiecaiMetr_score )    
