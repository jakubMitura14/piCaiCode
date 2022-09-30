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
    AsDiscreted,
    MapTransform,
    ResizeWithPadOrCropd,
    RepeatChanneld
    
)
from monai.config import KeysCollection
from monai.data import MetaTensor
import torchio
import numpy as np

class standardizeLabels(MapTransform):
    def __init__(
        self,
        keys: KeysCollection = "label",
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            d[key] = (d[key] > 0.5).astype('int8')
        return d

class wrapTorchio(MapTransform):
    def __init__(
        self,
        torchioObj,
        keys: KeysCollection = "chan3_col_name",
        # p: float=0.2,
        allow_missing_keys: bool = False,
        
    ):
        super().__init__(keys, allow_missing_keys)
        self.keys=keys
        self.torchioObj=torchioObj

    def __call__(self, data):
        return self.torchioObj(data)
        # d = dict(data)
        # for key in self.keys:
        #     d[key] = torchioObj()   (d[key] > 0.5).astype('int8')
        # return d




def decide_if_whole_image_train(is_whole_to_train, chan3Name,labelName):
    """
    if true we will trian on whole images otherwise just on 64x64x32
    randomly cropped parts
    """
    fff=not is_whole_to_train
    print(f" in decide_if_whole_image_train {fff}")
    if(not is_whole_to_train):
        return [RandCropByPosNegLabeld(
                keys=[chan3Name,labelName],
                label_key=labelName,
                spatial_size=(96, 96, 32),
                pos=1,
                neg=1,
                num_samples=2,
                image_key=chan3Name,
                image_threshold=0
            )
             ]
    return []         


#https://github.com/DIAGNijmegen/picai_prep/blob/19a0ef2d095471648a60e45e30b218b7a81b2641/src/picai_prep/preprocessing.py



def get_train_transforms(RandGaussianNoised_prob
    ,RandAdjustContrastd_prob
    ,RandGaussianSmoothd_prob
    ,RandRicianNoised_prob
    ,RandFlipd_prob
    ,RandAffined_prob
    ,RandCoarseDropoutd_prob
    ,is_whole_to_train
    ,centerCropSize 
    ,RandomElasticDeformation_prob
    ,RandomAnisotropy_prob
    ,RandomMotion_prob
    ,RandomGhosting_prob
    ,RandomSpike_prob
    ,RandomBiasField_prob):
    
    train_transforms = Compose(
        [
            LoadImaged(keys=["chan3_col_name","label"]),
            EnsureChannelFirstd(keys=["chan3_col_name","label"]),
            standardizeLabels(keys=["label"]),
            AsDiscreted(keys=["label"],to_onehot=2),
            #torchio.transforms.OneHot(include=["label"] ), #num_classes=3,
            #AsDiscreted(keys=["label"],to_onehot=2,threshold=0.5),
            #AsChannelFirstd(keys=["t2w","adc", "hbv","label"]),
            #Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
            #Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),            
            #CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),
            EnsureTyped(keys=["chan3_col_name","label"]),
            # SelectItemsd(keys=["chan3_col_name","label"]),
            #AddChanneld( keys=["chan3_col_name","label"]) ,          
            #ResizeWithPadOrCropd(keys=["chan3_col_name","label"],spatial_size=centerCropSize ),
          
            #*decide_if_whole_image_train(is_whole_to_train,"chan3_col_name","label"),
            #SpatialPadd(keys=["chan3_col_name","label"]],spatial_size=maxSize) ,            
            #RandGaussianNoised(keys=["chan3_col_name"], prob=RandGaussianNoised_prob),
            RandAdjustContrastd(keys=["chan3_col_name"], prob=RandAdjustContrastd_prob),
            RandGaussianSmoothd(keys=["chan3_col_name"], prob=RandGaussianSmoothd_prob),
            RandRicianNoised(keys=["chan3_col_name",], prob=RandRicianNoised_prob),
            RandFlipd(keys=["chan3_col_name","label"], prob=RandFlipd_prob),
            RandAffined(keys=["chan3_col_name","label"], prob=RandAffined_prob),
            #RandCoarseDropoutd(keys=["chan3_col_name"], prob=RandCoarseDropoutd_prob,holes=6, spatial_size=5),

            # wrapTorchio(torchio.transforms.RandomElasticDeformation(include=["chan3_col_name","label"],p=RandomElasticDeformation_prob)),
            # wrapTorchio(torchio.transforms.RandomAnisotropy(include=["chan3_col_name","label"],p=RandomAnisotropy_prob)),
            # wrapTorchio(torchio.transforms.RandomMotion(include=["chan3_col_name"],p=RandomMotion_prob)),
            # wrapTorchio(torchio.transforms.RandomGhosting(include=["chan3_col_name"],p=RandomGhosting_prob)),
            # wrapTorchio(torchio.transforms.RandomSpike(include=["chan3_col_name"],p=RandomSpike_prob)),
            # wrapTorchio(torchio.transforms.RandomBiasField(include=["chan3_col_name"],p=RandomBiasField_prob)),
            
            DivisiblePadd(keys=["chan3_col_name","label"],k=32)
            ,RandCropByPosNegLabeld(
            keys=["chan3_col_name","label"],
            label_key="label",
            spatial_size=(96, 96, 32),
            pos=1,
            neg=1,
            num_samples=2,
            image_key="chan3_col_name",
            image_threshold=0
        ),

         ]
    )
    return train_transforms
def get_val_transforms(is_whole_to_train,centerCropSize):
    val_transforms = Compose(
        [
            LoadImaged(keys=["chan3_col_name_val","label_name_val"]),
            EnsureChannelFirstd(keys=["chan3_col_name_val","label_name_val"]),
            standardizeLabels(keys=["label_name_val"]),
            AsDiscreted(keys=["label_name_val"], to_onehot=2),
            #AddChanneld( keys=["chan3_col_name","label"]) ,  
            #torchio.transforms.OneHot(include=["label"] ),#num_classes=3
            #AsDiscreted(keys=["label"],to_onehot=2,threshold=0.5),
            #AsChannelFirstd(keys=["chan3_col_name","label"]]),
            #Orientationd(keys=["chan3_col_name","label"]], axcodes="RAS"),
            # Spacingd(keys=["chan3_col_name","label"]], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            #SpatialPadd(keys=["chan3_col_name","label"],spatial_size=maxSize) ,
            DivisiblePadd(keys=["chan3_col_name_val","label_name_val"],k=32) ,
            #ResizeWithPadOrCropd(keys=["chan3_col_name","label_name_val"],spatial_size=centerCropSize ),

            #*decide_if_whole_image_train(is_whole_to_train,"chan3_col_name_val","label_name_val"),
            EnsureTyped(keys=["chan3_col_name_val","label_name_val"]),
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





