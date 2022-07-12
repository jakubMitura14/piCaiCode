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
    ConcatItemsd,
    RandCoarseDropoutd,
    AsDiscreted
    
)
import torchio


def decide_if_whole_image_train(is_whole_to_train):
    """
    if true we will trian on whole images otherwise just on 32x32x32
    randomly cropped parts
    """
    fff=not is_whole_to_train
    print(f" in decide_if_whole_image_train {fff}")
    if(not is_whole_to_train):
        return [DivisiblePadd(keys=["chan3_col_name"],k=32)
            ,RandCropByPosNegLabeld(
                keys=["chan3_col_name"],
                label_key="label",
                spatial_size=(32, 32, 32),
                pos=1,
                neg=1,
                num_samples=6,
                image_key="chan3_col_name",
                image_threshold=0,
            )
             ]
    return []         



def get_train_transforms(RandGaussianNoised_prob
    ,RandAdjustContrastd_prob
    ,RandGaussianSmoothd_prob
    ,RandRicianNoised_prob
    ,RandFlipd_prob
    ,RandAffined_prob
    ,RandCoarseDropoutd_prob
    ,is_whole_to_train ):
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["chan3_col_name","label"]),
            EnsureChannelFirstd(keys=["chan3_col_name","label"]),
            #torchio.transforms.OneHot(include=["label"] ), #num_classes=3,
            #AsDiscreted(keys=["label"],to_onehot=2,threshold=0.5),
            #AsChannelFirstd(keys=["t2w","adc", "hbv","label"]),
            #Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
            #Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),            
            #CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),
            EnsureTyped(keys=["chan3_col_name","label"]),
            # SelectItemsd(keys=["chan3_col_name","label"]),
            DivisiblePadd(keys=["chan3_col_name","label"],k=32) ,            
            *decide_if_whole_image_train(is_whole_to_train),
            #SpatialPadd(keys=["chan3_col_name","label"]],spatial_size=maxSize) ,            
            RandGaussianNoised(keys=["chan3_col_name"], prob=RandGaussianNoised_prob),
            RandAdjustContrastd(keys=["chan3_col_name"], prob=RandAdjustContrastd_prob),
            RandGaussianSmoothd(keys=["chan3_col_name"], prob=RandGaussianSmoothd_prob),
            RandRicianNoised(keys=["chan3_col_name",], prob=RandRicianNoised_prob),
            RandFlipd(keys=["chan3_col_name","label"], prob=RandFlipd_prob),
            RandAffined(keys=["chan3_col_name","label"], prob=RandAffined_prob),
            RandCoarseDropoutd(keys=["chan3_col_name"], prob=RandCoarseDropoutd_prob,holes=6, spatial_size=5),
            
        ]
    )
    return train_transforms
def get_val_transforms():
    val_transforms = Compose(
        [
            LoadImaged(keys=["chan3_col_name","label"]),
            EnsureChannelFirstd(keys=["chan3_col_name","label"]),
            #torchio.transforms.OneHot(include=["label"] ),#num_classes=3
            #AsDiscreted(keys=["label"],to_onehot=2,threshold=0.5),
            #AsChannelFirstd(keys=["chan3_col_name","label"]]),
            #Orientationd(keys=["chan3_col_name","label"]], axcodes="RAS"),
            # Spacingd(keys=["chan3_col_name","label"]], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            #SpatialPadd(keys=["chan3_col_name","label"],spatial_size=maxSize) ,
            DivisiblePadd(keys=["chan3_col_name","label"],k=32) ,
            #DivisiblePadd(keys=["chan3_col_name","label"],k=32) ,

            #CropForegroundd(keys=["chan3_col_name","label"]], source_key="image"),

            EnsureTyped(keys=["chan3_col_name","label"]),
            #SelectItemsd(keys=["chan3_col_name","label"]),
            # ConcatItemsd(keys=["t2w","adc","hbv"],name="chan3_col_name")
        ]
    )
    return val_transforms









# def decide_if_whole_image_train(is_whole_to_train,maxSize):
#     """
#     if true we will trian on whole images otherwise just on 32x32x32
#     randomly cropped parts
#     """
#     if(is_whole_to_train):
#         return [SpatialPadd(keys=["chan3_col_name","label"],spatial_size=maxSize)]
#     else:
#         return [DivisiblePadd(keys=["chan3_col_name","label"],k=32)
#             ,RandCropByPosNegLabeld(
#                 keys=["chan3_col_name","label"],
#                 label_key="label",
#                 spatial_size=(32, 32, 32),
#                 pos=1,
#                 neg=1,
#                 num_samples=4,
#                 image_key="chan3_col_name",
#                 image_threshold=0,
#             )
#              ]



# def get_train_transforms(maxSize
#     ,RandGaussianNoised_prob
#     ,RandAdjustContrastd_prob
#     ,RandGaussianSmoothd_prob
#     ,RandRicianNoised_prob
#     ,RandFlipd_prob
#     ,RandAffined_prob
#     ,RandCoarseDropoutd_prob
#     ,is_whole_to_train ):
    
#     train_transforms = Compose(
#         [
#             LoadImaged(keys=["t2w","adc", "hbv","label"]),
#             EnsureChannelFirstd(keys=["t2w","adc", "hbv","label"]),
#             #AsChannelFirstd(keys=["t2w","adc", "hbv","label"]),
#             #Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
#             #Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
#             #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),            
#             #CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),
#             EnsureTyped(keys=["t2w","adc", "hbv","label"]),
#             SelectItemsd(keys=["t2w","adc", "hbv","label"]),
#             ConcatItemsd(keys=["t2w","adc","hbv"],name="chan3_col_name"),
#             *decide_if_whole_image_train(is_whole_to_train,maxSize),
#             #SelectItemsd(keys=["chan3_col_name","adc","hbv","label"]),


#         #     #RandGaussianNoised(keys=["chan3_col_name"], prob=RandGaussianNoised_prob),
#         #     RandAdjustContrastd(keys=["chan3_col_name"], prob=RandAdjustContrastd_prob),
#         #     RandGaussianSmoothd(keys=["chan3_col_name"], prob=RandGaussianSmoothd_prob),
#         #     RandRicianNoised(keys=["chan3_col_name"], prob=RandRicianNoised_prob),
#         #     RandFlipd(keys=["chan3_col_name"], prob=RandFlipd_prob),
#         #     #RandAffined(keys=["chan3_col_name","label"], prob=RandAffined_prob),
#         #     RandCoarseDropoutd(keys=["chan3_col_name"], prob=RandCoarseDropoutd_prob,holes=6, spatial_size=5),
            


            
#         ]
#     )
#     return train_transforms
# def get_val_transforms(maxSize):
#     val_transforms = Compose(
#         [
#             LoadImaged(keys=["t2w","adc", "hbv","label"]),
#             EnsureChannelFirstd(keys=["t2w","adc", "hbv","label"]),
#             EnsureTyped(keys=["t2w","adc", "hbv","label"]),
#             SelectItemsd(keys=["t2w","adc", "hbv","label"]),
#             ConcatItemsd(keys=["t2w","adc","hbv"],name="chan3_col_name"),
#             SelectItemsd(keys=["chan3_col_name","label"]),
#             #AsChannelFirstd(keys=["t2w","adc", "hbv","label"]),
#             #Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
#             # Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
#             #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
#             SpatialPadd(keys=["chan3_col_name","label"],spatial_size=maxSize) ,
#             # DivisiblePadd(keys=["chan3_col_name","label"],k=32) ,
#             # SelectItemsd(keys=["chan3_col_name","adc","hbv","label"]),
            
#             #DivisiblePadd(keys=["t2w","adc", "hbv","label"],k=32) ,

#             #CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),

#             # SelectItemsd(keys=["t2w","adc", "hbv","label"]),
            
            

#         ]
#     )
#     return val_transforms





# def decide_if_whole_image_train(is_whole_to_train,maxSize):
#     """
#     if true we will trian on whole images otherwise just on 32x32x32
#     randomly cropped parts
#     """
#     if(is_whole_to_train):
#         return [SpatialPadd(keys=["chan3_col_name"],spatial_size=maxSize)]
#     else:
#         return [DivisiblePadd(keys=["chan3_col_name"],k=32)
#             ,RandCropByPosNegLabeld(
#                 keys=["chan3_col_name","label"],
#                 label_key="label",
#                 spatial_size=(32, 32, 32),
#                 pos=1,
#                 neg=1,
#                 num_samples=4,
#                 image_key="chan3_col_name",
#                 image_threshold=0,
#             )
#              ]

# def get_train_transforms(maxSize
#     ,RandGaussianNoised_prob
#     ,RandAdjustContrastd_prob
#     ,RandGaussianSmoothd_prob
#     ,RandRicianNoised_prob
#     ,RandFlipd_prob
#     ,RandAffined_prob
#     ,RandCoarseDropoutd_prob
#     ,is_whole_to_train ):
    
#     train_transforms = Compose(
#         [
#             LoadImaged(keys=["t2w","adc", "hbv","label"]),
#             SelectItemsd(keys=["t2w","adc", "hbv","label"]),
#             EnsureChannelFirstd(keys=["t2w","adc", "hbv","label"]),
#             #AsChannelFirstd(keys=["t2w","adc", "hbv","label"]),
#             #Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
#             #Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
#             #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),            
#             #CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),
#             EnsureTyped(keys=["t2w","adc", "hbv","label"]),
#             #DivisiblePadd(keys=["t2w","adc", "hbv","label"],k=32) ,

#             #SpatialPadd(keys=["t2w","adc", "hbv","label"],spatial_size=maxSize) ,            
#             RandGaussianNoised(keys=["t2w","adc", "hbv","label"], prob=RandGaussianNoised_prob),
#             RandAdjustContrastd(keys=["t2w","adc", "hbv","label"], prob=RandAdjustContrastd_prob),
#             RandGaussianSmoothd(keys=["t2w","adc", "hbv","label"], prob=RandGaussianSmoothd_prob),
#             RandRicianNoised(keys=["t2w","adc", "hbv","label"], prob=RandRicianNoised_prob),
#             RandFlipd(keys=["t2w","adc", "hbv","label"], prob=RandFlipd_prob),
#             RandAffined(keys=["t2w","adc", "hbv","label"], prob=RandAffined_prob),
#             RandCoarseDropoutd(keys=["t2w","adc", "hbv","label"], prob=RandCoarseDropoutd_prob,holes=6, spatial_size=5),
#             ConcatItemsd(keys=["t2w","adc","hbv"],name="chan3_col_name"),
#             *decide_if_whole_image_train(is_whole_to_train,maxSize),
#             #SelectItemsd(keys=["chan3_col_name","label"])

            
#         ]
#     )
#     return train_transforms
# def get_val_transforms(maxSize):
#     val_transforms = Compose(
#         [
#             LoadImaged(keys=["t2w","adc", "hbv","label"]),
#             SelectItemsd(keys=["t2w","adc", "hbv","label"]),
#             EnsureChannelFirstd(keys=["t2w","adc", "hbv","label"]),
#             #AsChannelFirstd(keys=["t2w","adc", "hbv","label"]),
#             #Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
#             # Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
#             #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
#             SpatialPadd(keys=["t2w","adc", "hbv","label"],spatial_size=maxSize) ,
#             #DivisiblePadd(keys=["t2w","adc", "hbv","label"],k=32) ,
#             DivisiblePadd(keys=["t2w","adc", "hbv","label"],k=32) ,

#             #CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),

#             EnsureTyped(keys=["t2w","adc", "hbv","label"]),
#             ConcatItemsd(keys=["t2w","adc","hbv"],name="chan3_col_name"),
#             #SelectItemsd(keys=["chan3_col_name","label"])

#         ]
#     )
#     return val_transforms





