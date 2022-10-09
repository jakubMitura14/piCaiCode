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
    EnsureChannelFirstd
    
)
from monai.config import KeysCollection



import torchio
import numpy as np

class standardizeLabels(MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        ref,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.ref=ref

    def __call__(self, data):

        d = dict(data)
        for key in self.keys:
            # print(f"in standd {d[key].meta}")
            d[key].set_array((d[key].get_array() > 0.5).astype('int8'))
            d[key].meta['pixdim']=d[self.ref].meta['pixdim']
            #update_meta(pixdim=d[self.ref].pixdim
            print(f" d[key].pixdim {d[key].pixdim} d[self.ref].pixdim {d[self.ref].pixdim} ")
        return d




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
    ,spatial_size
     ):


     
    train_transforms = Compose(
        [
            LoadImaged(keys=["t2w","hbv","adc" ,"label"]),
            EnsureTyped(keys=["t2w","hbv","adc" ,"label"]),
            EnsureChannelFirstd(keys=["t2w","hbv","adc" ,"label"]),
            standardizeLabels(keys=["label"],ref= "t2w"),
            Orientationd(keys=["t2w","adc", "hbv","label"], axcodes="RAS"),
            Spacingd(keys=["t2w","adc","hbv"], pixdim=(
                1.0, 1.0, 1.0), mode="bilinear" ),      #monai.utils.SplineMode.THREE
            Spacingd(keys=["label"], pixdim=(
                1.0, 1.0, 1.0), mode="nearest"),     
            # Spacingd(keys=["t2w","adc", "hbv","label"], pixdim=(
            #     1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            ResizeWithPadOrCropd(keys=["t2w","hbv","adc" ,"label"], spatial_size=spatial_size,),
            ConcatItemsd(keys=["t2w","adc","hbv","adc" ],name="chan3_col_name"),
            SelectItemsd(keys=["chan3_col_name","label","num_lesions_to_retain","isAnythingInAnnotated"]),
            AsDiscreted(keys=["label"],to_onehot=2),

            # standardizeLabels(keys=["label"]),
            # torchio.transforms.OneHot(include=["label"] ), #num_classes=3,
            #AsDiscreted(keys=["label"],to_onehot=2,threshold=0.5),
            #AsChannelFirstd(keys=["t2w","adc", "hbv","label"]),
       
            #CropForegroundd(keys=["t2w","adc", "hbv","label"], source_key="image"),
            # SelectItemsd(keys=["chan3_col_name","label"]),
            #DivisiblePadd(keys=["chan3_col_name","label"],k=32) ,            
            #ResizeWithPadOrCropd(keys=["chan3_col_name","label"],spatial_size=centerCropSize ),
          
            #*decide_if_whole_image_train(False,"chan3_col_name","label"),
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
def get_val_transforms(is_whole_to_train,spatial_size):
    print(f"spatial_sizeeee {spatial_size}"  )

    val_transforms = Compose(
        [
            LoadImaged(keys=["t2w","hbv","adc" ,"label_name_val"]),
            EnsureChannelFirstd(keys=["t2w","hbv","adc" ,"label_name_val"]),
            EnsureTyped(keys=["t2w","hbv","adc" ,"label_name_val"]),
            Orientationd(keys=["t2w","adc", "hbv","label_name_val"], axcodes="RAS"),
            standardizeLabels(keys=["label_name_val"],ref= "t2w"),
            Spacingd(keys=["t2w","adc","hbv"], pixdim=(
                1.0, 1.0, 1.0), mode="bilinear"),      
            Spacingd(keys=["label_name_val"], pixdim=(
                1.0, 1.0, 1.0), mode="nearest"),  
            ConcatItemsd(["t2w","label_name_val","hbv","adc" ], "dummy"),

            ResizeWithPadOrCropd(keys=["t2w","hbv","adc" ,"label_name_val"], spatial_size=spatial_size,),
            ConcatItemsd(["t2w","adc","hbv","adc" ], "chan3_col_name_val"),
            AsDiscreted(keys=["label_name_val"], to_onehot=2),

            #torchio.transforms.OneHot(include=["label"] ),#num_classes=3
            #AsChannelFirstd(keys=["chan3_col_name","label"]]),
            # Orientationd(keys=["chan3_col_name","label_name_val"], axcodes="RAS"),
            # Spacingd(keys=["chan3_col_name","label_name_val"], pixdim=(
            #     1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            #SpatialPadd(keys=["chan3_col_name","label"],spatial_size=maxSize) ,
            #DivisiblePadd(keys=["chan3_col_name_val","label_name_val"],k=32) ,
            #ResizeWithPadOrCropd(keys=["chan3_col_name","label_name_val"],spatial_size=centerCropSize ),

            #*decide_if_whole_image_train(is_whole_to_train,"chan3_col_name_val","label_name_val"),
            SelectItemsd(keys=["chan3_col_name_val","label_name_val","study_id","num_lesions_to_retain","isAnythingInAnnotated"]),
            # ConcatItemsd(keys=["t2w","adc","hbv"],name="chan3_col_name")
        ]
    )
    return val_transforms





