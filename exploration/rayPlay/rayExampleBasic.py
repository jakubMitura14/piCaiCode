"""Simple example using RayAccelerator and Ray Tune"""
import functools
import glob
import importlib.util
import math
import multiprocessing as mp
import operator
import os
import shutil
import sys
import tempfile
import time
import warnings
from datetime import datetime
from functools import partial
from glob import glob
from os import path as pathOs
from os.path import basename, dirname, exists, isdir, join, split
from pathlib import Path
#from picai_eval.picai_eval import evaluate_case
from statistics import mean
from typing import List, Optional, Sequence, Tuple, Union

import gdown
import matplotlib.pyplot as plt
import monai
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import ray
import seaborn as sns
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio
import torchio as tio
import torchmetrics
from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule
from pytorch_lightning import (Callback, LightningDataModule, LightningModule,
                               Trainer)
from pytorch_lightning.strategies import Strategy
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback, TuneReportCheckpointCallback)
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray_lightning import RayShardedStrategy, RayStrategy
from ray_lightning.tune import TuneReportCallback, get_tune_resources
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from torch.nn.intrinsic.qat import ConvBnReLU3d
from torch.utils.cpp_extension import load
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import Precision
from torchmetrics.functional import precision_recall
from ray.tune.schedulers.pb2 import PB2

ray.init(num_cpus=24)
data_dir = '/home/sliceruser/mnist'
#MNISTDataModule(data_dir=data_dir).prepare_data()
num_cpus_per_worker=6
test_l_dir = '/home/sliceruser/test_l_dir'

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

class netB(nn.Module):
    def __init__(self,
        config
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
        torch.nn.Linear(10, 24),
        torch.nn.Linear(24, 100),    
        torch.nn.Linear(100, 10)
        )
    def forward(self, x):
        return self.model(x)

class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()

        self.data_dir = data_dir or os.getcwd()
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]

        self.accuracy = torchmetrics.Accuracy()
        self.netA= netaA(config)
        self.netB= netB(config)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x= self.netA(x)
        x= self.netB(x)

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

    model = LightningMNISTClassifier(config, data_dir)




    callbacks = callbacks or []
    print(" aaaaaaaaaa  ")
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
        strategy=RayStrategy(
            num_workers=num_workers, use_gpu=use_gpu)
            
            )#, init_hook=download_data

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
        "lr": 1e-3,
        "batch_size": tune.choice([32, 64, 128]),
    }

    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
   
   #***********************************************
    #do not work
    #callbacks = [TuneReportCheckpointCallback(metrics, on="validation_end",filename="checkpointtt")]
    
    #works
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
 
    #***********************************************
    pb2_scheduler = PB2(
        time_attr="training_iteration",
        metric='acc',
        mode='max',
        perturbation_interval=10.0,
        hyperparam_bounds={
            "lr": [1e-2, 1e-5],
        })
 
 
    trainable = tune.with_parameters(
        train_mnist,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=callbacks)
    analysis = tune.run(
        trainable,
        scheduler=pb2_scheduler,
        # metric="loss",
        #mode="max",
        config=config,
        num_samples=num_samples,
        resources_per_trial=get_tune_resources(
            num_workers=num_workers, use_gpu=use_gpu),
        name="tune_mnist")

    print("Best hyperparameters found were: ", analysis.best_config)

tune_mnist(data_dir)
