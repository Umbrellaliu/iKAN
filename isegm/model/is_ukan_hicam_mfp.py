
import numpy as np
import torch.nn as nn
import torch

from isegm.utils.serialization import serialize


from .is_model_hicam_fusion_mfp import ISModel_hicam_fusion_mfp


from isegm.model.modifiers import LRMult
import matplotlib.pyplot as plt

from .modeling.ukan_arch_resnestfusion import *




class UKANModel_hicam_fusion_mfp(ISModel_hicam_fusion_mfp):
    @serialize
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3, embed_dims=[256, 320, 512], no_kan=False,
    drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[1, 1, 1], backbone_lr_mult=0.1,**kwargs):
        super().__init__(**kwargs)

        self.feature_extractor = UKAN_resnet_fusion(num_classes=num_classes, input_channels=input_channels, deep_supervision=deep_supervision, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dims=embed_dims, no_kan=no_kan,
    drop_rate=drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer, depths=depths)
        self.feature_extractor.apply(LRMult(backbone_lr_mult))



    def backbone_forward(self, image, coord_features=None):
        net_outputs = self.feature_extractor(image, coord_features)

        return {'instances': net_outputs, 'instances_aux': None}

