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
from pytorch_lightning.loggers import WandbLogger

sns.set()
plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()
from monai.transforms import (
    EnsureChannelFirstd,
    Orientationd,
    AsDiscrete,
    AddChanneld,
    Spacingd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Resize,
    Resized,
    RandSpatialCropd
)
print("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz")
######################################################3
from datetime import datetime
import os
import tempfile
from glob import glob

import torch
from torch.utils.data import random_split, DataLoader
import monai
import pandas as pd
import torchio as tio
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
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


sns.set()
plt.rcParams['figure.figsize'] = 12, 8
monai.utils.set_determinism()


#wandb_logger = WandbLogger(project="picai", name = "baseline")



df = pd.read_csv('/home/sliceruser/labels/processedMetaData.csv')

df = df.loc[df['isAnyMissing'] ==False]
df = df.loc[df['isAnythingInAnnotated']>0 ]

def getMonaiSubjectDataFromDataFrame(row):
        """
        given row from data frame prepares Subject object from it
        """
        subject= {"adc": str(row['adc'])        
        , "cor":str(row['cor'])
        , "hbv":str(row['hbv'])
        , "sag":str(row['sag'])
        , "t2w":str(row['t2w'])
        , "isAnythingInAnnotated":row['isAnythingInAnnotated']
        , "patient_id":row['patient_id']
        , "study_id":row['study_id']
        , "patient_age":row['patient_age']
        , "psa":row['psa']
        , "psad":row['psad']
        , "prostate_volume":row['prostate_volume']
        , "histopath_type":row['histopath_type']
        , "lesion_GS":row['lesion_GS']
        , "label":str(row['reSampledPath'])
        
        
        }

        return subject


data_dicts = list(map(lambda row: getMonaiSubjectDataFromDataFrame(row[1])  , list(df.iterrows())))


train_files, val_files = data_dicts[:-9], data_dicts[-9:]

set_determinism(seed=0)

train_transforms = Compose(
    [
        LoadImaged(keys=["t2w", "label"]),
        EnsureChannelFirstd(keys=["t2w", "label"]),
        Orientationd(keys=["t2w", "label"], axcodes="RAS"),
        Spacingd(keys=["t2w", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        tio.transforms.EnsureShapeMultiple((32 , 32, 32), include=["t2w", "label"]),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["t2w", "label"],
            label_key="label",
            spatial_size=(32, 32, 32),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="t2w",
            image_threshold=0,
        ),
        EnsureTyped(keys=["t2w", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["t2w", "label"]),
        EnsureChannelFirstd(keys=["t2w", "label"]),
        Orientationd(keys=["t2w", "label"], axcodes="RAS"),
        Spacingd(keys=["t2w", "label"], pixdim=(
            1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        tio.transforms.EnsureShapeMultiple((32 , 32, 32), include=["t2w", "label"]),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        # RandCropByPosNegLabeld(
        #     keys=["t2w", "label"],
        #     label_key="label",
        #     spatial_size=(32, 32, 32),
        #     pos=1,
        #     neg=1,
        #     num_samples=4,
        #     image_key="t2w",
        #     image_threshold=0,
        # ),
        EnsureTyped(keys=["t2w", "label"]),
    ]
)

### Get Rid off cases where transforms lead to error for any reason

deficientPatIDs=[]
for dictt in data_dicts:    
    try:
        dat = train_transforms(dictt)
        dat = val_transforms(dictt)
    except:
        deficientPatIDs.append(dictt['patient_id'])
        print(dictt['patient_id'])


def isInDeficienList(row):
        return row['patient_id'] not in deficientPatIDs

df["areTransformsNotDeficient"]= df.apply(lambda row : isInDeficienList(row), axis = 1)  

df = df.loc[ df['areTransformsNotDeficient']]





### Define Data Handling
class PiCaiDataModule(pl.LightningDataModule):
    def __init__(self,trainSizePercent,batch_size,num_workers,drop_last,df):
        super().__init__()
        self.batch_size = batch_size
        self.df = df
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.train_set = None
        self.val_set = None
        self.test_set = None  
        self.trainSizePercent =trainSizePercent
        self.train_files = None
        self.val_files= None
        self.test_files= None
        self.train_ds = None
        self.val_ds= None
        self.test_ds= None        
        self.subjects= None        

    def splitDataSet(self,patList, trainSizePercent):
        """
        test train validation split
        TODO(balance sets)
        """
        totalLen=len(patList)
        train_test_split( patList  )
        numTrain= math.ceil(trainSizePercent*totalLen)
        numTestAndVal=totalLen-numTrain
        numTest=math.ceil(numTestAndVal*0.5)
        numVal= numTestAndVal-numTest

        # valid_set,test_set = torch.utils.data.random_split(test_and_val_set, [math. ceil(0.5), 0.5])
        print('Train data set:', numTrain)
        print('Test data set:',numTest)
        print('Valid data set:', numVal)
        return torch.utils.data.random_split(patList, [numTrain,numVal,numTest])

    def get_preprocessing_transform(self):
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.CropOrPad(self.get_max_shape(self.subjects + self.test_subjects)),
            tio.EnsureShapeMultiple(8),  # for the U-Net
            tio.OneHot(),
        ])
        return preprocess
    
    def get_augmentation_transform(self):
        augment = tio.Compose([
            tio.RandomAffine(),
            tio.RandomGamma(p=0.5),
            tio.RandomNoise(p=0.5),
            tio.RandomMotion(p=0.1),
            tio.RandomBiasField(p=0.25),
        ])
        return augment

    def setup(self, stage=None):
        set_determinism(seed=0)
        self.subjects = list(map(lambda row: getMonaiSubjectDataFromDataFrame(row[1])   , list(df.iterrows())))
        train_set, valid_set,test_set = self.splitDataSet(self.subjects , self.trainSizePercent)
        self.train_subjects = train_set
        self.val_subjects = valid_set
        self.test_subjects = test_set
        self.train_ds =  Dataset(data=self.train_subjects, transform=train_transforms)
        self.val_ds=     Dataset(data=self.val_subjects, transform=val_transforms)
        self.test_ds=    Dataset(data=self.test_subjects, transform=val_transforms)    

    def train_dataloader(self):
        return DataLoader(self.train_ds, self.batch_size, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.val_ds, self.batch_size, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, self.batch_size, drop_last=self.drop_last)#num_workers=self.num_workers,

data = PiCaiDataModule(
    df= df,
    batch_size=1,#TODO(batc size determined by lightning)
    trainSizePercent=0.7,
    num_workers=1,
    drop_last=True
)
data.prepare_data()
data.setup()

class Model(pl.LightningModule):
    def __init__(self, net, criterion, learning_rate, optimizer_class):
        super().__init__()
        self.lr = learning_rate
        self.net = net
        self.criterion = criterion
        self.optimizer_class = optimizer_class
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        return optimizer
    
    def prepare_batch(self, batch):
        return batch['t2w'], batch['label']
    
    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def training_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat, y = self.infer_batch(batch)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss


unet = monai.networks.nets.UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    #channels=(16, 16, 16, 16, 16),
    strides=(2, 2, 2, 2),
    num_res_units=1,
)

model = Model(
    net=unet,
    criterion=monai.losses.DiceCELoss(softmax=True),
    learning_rate=1e-2,
    optimizer_class=torch.optim.AdamW,
)
early_stopping = pl.callbacks.early_stopping.EarlyStopping(
    monitor='val_loss',
)
trainer = pl.Trainer(
    #accelerator="cpu", #TODO(remove)
    max_epochs=5,
    #gpus=1,
    #precision=16, #TODO(unhash)
    callbacks=[early_stopping],
    #logger=wandb_logger,
    accelerator='gpu',
    devices=1
)
trainer.logger._default_hp_metric = False


start = datetime.now()
print('Training started at', start)
trainer.fit(model=model, datamodule=data)
print('Training duration:', datetime.now() - start)

# # for idx, (data) in enumerate(train_ds):
# #     print(idx)

# for epoch in range(max_epochs):
#     print("-" * 10)
#     print(f"epoch {epoch + 1}/{max_epochs}")
#     model.train()
#     epoch_loss = 0
#     step = 0
#     for batch_data in train_loader:
#         step += 1
#         inputs, labels = (
#             batch_data["t2w"].to(device),
#             batch_data["label"].to(device),
#         )
#         optimizer.zero_grad()

#         # print("batch_data[image]")
#         # print(batch_data["image"].shape)

#         # print("batch_data[label]")
#         # print(batch_data["label"].shape)

#         outputs = model(inputs)
#         loss = loss_function(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         # print("epoch_loss")
#         # print(epoch_loss)
#         # print("loss.item()")
#         # print(loss.item())
#         epoch_loss += loss.item()
#         # print(
#         #     f"{step}/{len(train_ds) // train_loader.batch_size}, "
#         #     f"train_loss: {loss.item():.4f}")
#     epoch_loss /= step
#     epoch_loss_values.append(epoch_loss)
#     print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

#     #if (epoch + 1) % val_interval == 0:
#     if (True): #TODO(just for debugging)
#         model.eval()
#         with torch.no_grad():
#             for val_data in val_loader:
#                 val_inputs, val_labels = (
#                     val_data["t2w"].to(device),
#                     val_data["label"].to(device),
#                 )
#                 roi_size = (8, 8, 8)
#                 sw_batch_size = 1
#                 # model.eval()# as in https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/2
#                 print("vvvv val_inputs vvvvvv")
#                 print(val_inputs.shape)
                
#                 val_outputs =model(val_inputs)
                
#                 #  sliding_window_inference(
#                 #     val_inputs, roi_size, sw_batch_size, model)

#                 val_outputs = [i for i in decollate_batch(val_outputs)]
#                 val_labels = [i for i in decollate_batch(val_labels)]
                
#                 # compute metric for current iteration
#                 dice_metric(y_pred=val_outputs, y=val_labels)

#             # aggregate the final mean dice result
#             metric = dice_metric.aggregate().item()
#             # reset the status for next validation round
#             dice_metric.reset()

#             metric_values.append(metric)
#             if metric > best_metric:
#                 best_metric = metric
#                 best_metric_epoch = epoch + 1
#                 torch.save(model.state_dict(), os.path.join(
#                     root_dir, "best_metric_model.pth"))
#                 print("saved new best metric model")
#             print(
#                 f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
#                 f"\nbest mean dice: {best_metric:.4f} "
#                 f"at epoch: {best_metric_epoch}"
#             )
# print(
#     f"train completed, best_metric: {best_metric:.4f} "
#     f"at epoch: {best_metric_epoch}")




