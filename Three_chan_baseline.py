from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
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

sns.set()
plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()

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

import torch.nn as nn
import torch.nn.functional as F

import multiprocessing

sns.set()
plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()
# import preprocessing.transformsForMain as transformsForMain
# import preprocessing.ManageMetadata as manageMetaData
# import dataManag.utils.dataUtils as dataUtils
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("transformsForMain", "/home/sliceruser/data/piCaiCode/preprocessing/transformsForMain.py")
transformsForMain = importlib.util.module_from_spec(spec)
sys.modules["transformsForMain"] = transformsForMain
spec.loader.exec_module(transformsForMain)

spec = importlib.util.spec_from_file_location("ManageMetadata", "/home/sliceruser/data/piCaiCode/preprocessing/ManageMetadata.py")
manageMetaData = importlib.util.module_from_spec(spec)
sys.modules["ManageMetadata"] = manageMetaData
spec.loader.exec_module(manageMetaData)


spec = importlib.util.spec_from_file_location("dataUtils", "/home/sliceruser/data/piCaiCode/dataManag/utils/dataUtils.py")
dataUtils = importlib.util.module_from_spec(spec)
sys.modules["dataUtils"] = dataUtils
spec.loader.exec_module(dataUtils)


spec = importlib.util.spec_from_file_location("unets", "/home/sliceruser/data/piCaiCode/model/unets.py")
unets = importlib.util.module_from_spec(spec)
sys.modules["unets"] = unets
spec.loader.exec_module(unets)


spec = importlib.util.spec_from_file_location("DataModule", "/home/sliceruser/data/piCaiCode/model/DataModule.py")
DataModule = importlib.util.module_from_spec(spec)
sys.modules["DataModule"] = DataModule
spec.loader.exec_module(DataModule)

spec = importlib.util.spec_from_file_location("LigtningModel", "/home/sliceruser/data/piCaiCode/model/LigtningModel.py")
LigtningModel = importlib.util.module_from_spec(spec)
sys.modules["LigtningModel"] = LigtningModel
spec.loader.exec_module(LigtningModel)





comet_logger = CometLogger(
    api_key="yB0irIjdk9t7gbpTlSUPnXBd4",
    #workspace="OPI", # Optional
    project_name="picai_base_3Channels", # Optional
    #experiment_name="baseline" # Optional
)

df = pd.read_csv('/home/sliceruser/data/metadata/processedMetaData_current.csv')
maxSize=manageMetaData.getMaxSize("t2w_med_spac",df)
df= manageMetaData.load_df_only_full(df,"t2w_med_spac","registered_adc_med_spac","registered_hbv_med_spac", "label_med_spac",maxSize )


data = DataModule.PiCaiDataModule(
    df= df,
    batch_size=3,#TODO(batc size determined by lightning)
    trainSizePercent=0.7,
    num_workers=os.cpu_count(),
    drop_last=False,#True,
    cache_dir="/home/sliceruser/preprocess/monai_persistent_Dataset",
    t2w_name="t2w_med_spac",
    adc_name="registered_adc_med_spac",
    hbv_name="registered_hbv_med_spac",
    label_name="label_med_spac",
    maxSize=maxSize
)
data.prepare_data()
data.setup()

# definition described in model folder
# from https://github.com/DIAGNijmegen/picai_baseline/blob/main/src/picai_baseline/unet/training_setup/neural_networks/unets.py
unet= unets.UNet(
    spatial_dims=3,
    in_channels=3,
    out_channels=2,
    strides=[(2, 2, 2), (1, 2, 2), (1, 2, 2), (1, 2, 2), (2, 2, 2)],
    channels=[32, 64, 128, 256, 512, 1024]
)


model = LigtningModel.Model(
    net=unet,
    criterion=monai.losses.DiceCELoss(softmax=True,onehot_y=True), # Our seg labels are single channel images indicating class index, rather than one-hot
    learning_rate=1e-2,
    optimizer_class=torch.optim.AdamW,
)
early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss',
)
trainer = pl.Trainer(
    #accelerato="cpu", #TODO(remove)
    max_epochs=3,
    #gpus=1,
    precision=16, #TODO(unhash)
    callbacks=[early_stopping],#TODO(unhash)
    logger=comet_logger,
    accelerator='auto',
    devices='auto',
    default_root_dir= "/home/sliceruser/lightning_logs",
)
trainer.logger._default_hp_metric = False


start = datetime.now()
print('Training started at', start)
trainer.fit(model=model, datamodule=data)
print('Training duration:', datetime.now() - start)
#experiment.end()
