import monai
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
    RandSpatialCropd,
        AsDiscrete,
    AsDiscreted,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    SaveImaged,
    ScaleIntensityRanged,
    SelectItemsd,
    Invertd,
    DivisiblePadd,
    SpatialPadd,
    RandGaussianNoised,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandRicianNoised,
    RandFlipd,
    RandAffined,
    ConcatItemsd
    
)
import torch
import torchio as tio


def get_train_transforms(maxSize):
    train_transforms = Compose(
        [
            LoadImaged(keys=["t2w","adc", "hbv","label"]),
            EnsureChannelFirstd(keys=["t2w","adc", "hbv","label"]),
            #AsChannelFirstd(keys=["t2w","adc", "hbv","label"]),
            #Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
            #Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),            
            #CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),
            EnsureTyped(keys=["t2w","adc", "hbv","label"]),
            SelectItemsd(keys=["t2w","adc", "hbv","label"]),
            DivisiblePadd(keys=["t2w","adc", "hbv","label"],k=32) ,

            #SpatialPadd(keys=["t2w","adc", "hbv","label"],spatial_size=maxSize) ,            
            RandGaussianNoised(keys=["t2w","adc", "hbv","label"], prob=0.1),
            RandAdjustContrastd(keys=["t2w","adc", "hbv","label"], prob=0.1),
            RandGaussianSmoothd(keys=["t2w","adc", "hbv","label"], prob=0.1),
            RandRicianNoised(keys=["t2w","adc", "hbv","label"], prob=0.1),
            RandFlipd(keys=["t2w","adc", "hbv","label"], prob=0.1),
            RandAffined(keys=["t2w","adc", "hbv","label"], prob=0.1),
            ConcatItemsd(keys=["t2w","adc","hbv"],name="common_3channels"),
            RandCropByPosNegLabeld(
                keys=["common_3channels","label"],
                label_key="label",
                spatial_size=(32, 32, 32),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="common_3channels",
                image_threshold=0,
            ),            
            
        ]
    )
    return train_transforms
def get_val_transforms(maxSize):
    val_transforms = Compose(
        [
            LoadImaged(keys=["t2w","adc", "hbv","label"]),
            EnsureChannelFirstd(keys=["t2w","adc", "hbv","label"]),
            #AsChannelFirstd(keys=["t2w","adc", "hbv","label"]),
            #Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
            # Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            #SpatialPadd(keys=["t2w","adc", "hbv","label"],spatial_size=maxSize) ,
            #DivisiblePadd(keys=["t2w","adc", "hbv","label"],k=32) ,
            DivisiblePadd(keys=["t2w","adc", "hbv","label"],k=32) ,

            #CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),

            EnsureTyped(keys=["t2w","adc", "hbv","label"]),
            SelectItemsd(keys=["t2w","adc", "hbv","label"]),
            ConcatItemsd(keys=["t2w","adc","hbv"],name="common_3channels")
        ]
    )
    return val_transforms







# def get_train_transforms():
#     train_transforms = Compose(
#         [
#             LoadImaged(keys=["t2w","adc", "hbv","label"]),
#            Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
#             tio.transforms.EnsureShapeMultiple((32 , 32, 32), include=["t2w","adc", "hbv","label"]),           
#            # AddChanneld(keys=["t2w","adc", "hbv","label"]),
#             EnsureChannelFirstd(keys=["t2w","adc", "hbv","label"]),
#             Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
#                 1.5, 1.5, 2.0), mode=("bilinear", "nearest")),


#             CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),
#             RandCropByPosNegLabeld(
#                 keys=["t2w","adc", "hbv","label"],
#                 label_key="label",
#                 spatial_size=(32, 32, 32),
#                 pos=1,
#                 neg=1,
#                 num_samples=4,
#                 image_key="t2w",
#                 image_threshold=0,
#             ),
#             EnsureTyped(keys=["t2w","adc", "hbv","label"]),
#             SelectItemsd(keys=["t2w","adc", "hbv","label"])#TODO remove
#         ]
#     )
#     return train_transforms
# def get_val_transforms():
#     val_transforms = Compose(
#         [
#             LoadImaged(keys=["t2w","adc", "hbv","label"]),
#             tio.transforms.EnsureShapeMultiple((32 , 32, 32), include=["t2w","adc", "hbv","label"]),            
#            Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),            
#            # AddChanneld(keys=["t2w","adc", "hbv","label"]),            
#             EnsureChannelFirstd(keys=["t2w","adc", "hbv","label"]),
#             Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
#                 1.5, 1.5, 2.0), mode=("bilinear", "nearest")),

#             CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),
#             RandCropByPosNegLabeld(
#                 keys=["t2w","adc", "hbv","label"],
#                 label_key="label",
#                 spatial_size=(32, 32, 32),
#                 pos=1,
#                 neg=1,
#                 num_samples=4,
#                 image_key="t2w",
#                 image_threshold=0,
#             ),
#             EnsureTyped(keys=["t2w","adc", "hbv","label"]),
#             SelectItemsd(keys=["t2w","adc", "hbv","label"]) #TODO remove
#         ]
#     )
#     return val_transforms
