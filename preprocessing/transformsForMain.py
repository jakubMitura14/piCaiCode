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
    SpatialPadd
)
import torch
import torchio as tio





def get_train_transforms():
    train_transforms = Compose(
        [
            LoadImaged(keys=["t2w", "label"]),
            EnsureChannelFirstd(keys=["t2w", "label"]),
            #AsChannelFirstd(keys=["t2w", "label"]),
            Orientationd(keys=["t2w", "label"], axcodes="RAS"),
            Spacingd(keys=["t2w", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            DivisiblePadd(keys=["t2w", "label"],k=32) ,
            #CropForegroundd(keys=["t2w", "label"], source_key="image"),
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
            SelectItemsd(keys=["t2w", "label"])
        ]
    )
    return train_transforms
def get_val_transforms():
    val_transforms = Compose(
        [
            LoadImaged(keys=["t2w", "label"]),
            EnsureChannelFirstd(keys=["t2w", "label"]),
            #AsChannelFirstd(keys=["t2w", "label"]),
            Orientationd(keys=["t2w", "label"], axcodes="RAS"),
            Spacingd(keys=["t2w", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            #SpatialPadd(keys=["t2w", "label"],spatial_size=(1024,1024,64)) ,
            DivisiblePadd(keys=["t2w", "label"],k=32) ,

            #CropForegroundd(keys=["t2w", "label"], source_key="image"),

            EnsureTyped(keys=["t2w", "label"]),
            SelectItemsd(keys=["t2w", "label"])
        ]
    )
    return val_transforms






# def get_train_transforms():
#     train_transforms = Compose(
#         [
#             LoadImaged(keys=["t2w", "label"]),
#            Orientationd(keys=["t2w", "label"], axcodes="RAS"),
#             tio.transforms.EnsureShapeMultiple((32 , 32, 32), include=["t2w", "label"]),           
#            # AddChanneld(keys=["t2w", "label"]),
#             EnsureChannelFirstd(keys=["t2w", "label"]),
#             Spacingd(keys=["t2w", "label"], pixdim=(
#                 1.5, 1.5, 2.0), mode=("bilinear", "nearest")),


#             CropForegroundd(keys=["t2w", "label"], source_key="image"),
#             RandCropByPosNegLabeld(
#                 keys=["t2w", "label"],
#                 label_key="label",
#                 spatial_size=(32, 32, 32),
#                 pos=1,
#                 neg=1,
#                 num_samples=4,
#                 image_key="t2w",
#                 image_threshold=0,
#             ),
#             EnsureTyped(keys=["t2w", "label"]),
#             SelectItemsd(keys=["t2w", "label"])#TODO remove
#         ]
#     )
#     return train_transforms
# def get_val_transforms():
#     val_transforms = Compose(
#         [
#             LoadImaged(keys=["t2w", "label"]),
#             tio.transforms.EnsureShapeMultiple((32 , 32, 32), include=["t2w", "label"]),            
#            Orientationd(keys=["t2w", "label"], axcodes="RAS"),            
#            # AddChanneld(keys=["t2w", "label"]),            
#             EnsureChannelFirstd(keys=["t2w", "label"]),
#             Spacingd(keys=["t2w", "label"], pixdim=(
#                 1.5, 1.5, 2.0), mode=("bilinear", "nearest")),

#             CropForegroundd(keys=["t2w", "label"], source_key="image"),
#             RandCropByPosNegLabeld(
#                 keys=["t2w", "label"],
#                 label_key="label",
#                 spatial_size=(32, 32, 32),
#                 pos=1,
#                 neg=1,
#                 num_samples=4,
#                 image_key="t2w",
#                 image_threshold=0,
#             ),
#             EnsureTyped(keys=["t2w", "label"]),
#             SelectItemsd(keys=["t2w", "label"]) #TODO remove
#         ]
#     )
#     return val_transforms
