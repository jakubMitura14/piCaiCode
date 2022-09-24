"""Simple example using RayAccelerator and Ray Tune"""
import os
import tempfile

import time
from pathlib import Path
from datetime import datetime
import SimpleITK as sitk
from monai.utils import set_determinism
import math
import torch
from torch.utils.data import random_split, DataLoader
import monai
import gdown
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset,Dataset,PersistentDataset, list_data_collate, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract

from datetime import datetime
import os
import tempfile
from glob import glob
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from picai_eval import evaluate
#from picai_eval.picai_eval import evaluate_case
from statistics import mean
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter
import tempfile
import shutil
from os import path as pathOs
from os.path import basename, dirname, exists, isdir, join, split
import torch.nn as nn
import torch.nn.functional as F

from monai.metrics import (DiceMetric, HausdorffDistanceMetric,
                           SurfaceDistanceMetric)
from torchmetrics import Precision
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
)
import torchio
import importlib.util
import sys
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

import multiprocessing as mp
import time
from functools import partial
from torchmetrics.functional import precision_recall
from torch.utils.cpp_extension import load
import torchmetrics
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

import pytorch_lightning as pl
import ray
from ray import tune
from ray_lightning.tune import TuneReportCallback, get_tune_resources
from ray_lightning import RayStrategy
import os
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning.strategies import Strategy
from pytorch_lightning import LightningModule, Callback, Trainer, \
    LightningDataModule

import torchmetrics


ray.init(num_cpus=24)
data_dir = '/home/sliceruser/mnist'
#MNISTDataModule(data_dir=data_dir).prepare_data()
num_cpus_per_worker=6

class netaA(nn.Module):
    def __init__(self,
        config
    ) -> None:
        super().__init__()
        layer_1, layer_2 = config["layer_1"], config["layer_2"]
        self.model = nn.Sequential(
        torch.nn.Linear(28 * 28, layer_1),
        torch.nn.Linear(layer_1, layer_2),    
        torch.nn.Linear(layer_2, 10)
        )
    def forward(self, x):
        return self.model(x)



class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()

        self.data_dir = data_dir or os.getcwd()
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]

        # mnist images are (1, 28, 28) (channels, width, height)
        # self.layer_1 = torch.nn.Linear(28 * 28, layer_1)
        # self.layer_2 = torch.nn.Linear(layer_1, layer_2)
        # self.layer_3 = torch.nn.Linear(layer_2, 10)
        self.accuracy = torchmetrics.Accuracy()
        self.netA= netaA(config)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x= self.netA(x)

        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        acc = self.accuracy(logits, y)
        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        acc = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)



def train_mnist(config,
                data_dir=None,
                num_epochs=10,
                num_workers=1,
                use_gpu=True,
                callbacks=None):
    # Make sure data is downloaded on all nodes.
    # def download_data():
    #     from filelock import FileLock
    #     with FileLock(os.path.join(data_dir, ".lock")):
    #         MNISTDataModule(data_dir=data_dir).prepare_data()

    model = LightningMNISTClassifier(config, data_dir)

    callbacks = callbacks or []
    print(" aaaaaaaaaa  ")
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
        strategy=RayStrategy(
            num_workers=num_workers, use_gpu=use_gpu))#, init_hook=download_data
    dm = MNISTDataModule(
        data_dir=data_dir, num_workers=2, batch_size=config["batch_size"])
    trainer.fit(model, dm)


def tune_mnist(data_dir,
               num_samples=2,
               num_epochs=10,
               num_workers=2,
               use_gpu=True):
    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    # Add Tune callback.
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
    trainable = tune.with_parameters(
        train_mnist,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=callbacks)
    analysis = tune.run(
        trainable,
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        resources_per_trial=get_tune_resources(
            num_workers=num_workers, use_gpu=use_gpu),
        name="tune_mnist")

    print("Best hyperparameters found were: ", analysis.best_config)

tune_mnist(data_dir)
