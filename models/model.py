import torch
import torch.nn as nn
from torch.nn import functional as F

from backbone import Backbone
from layers import GlobalDescriptor


class Model(nn.Module):

    def __init__(self,
                 gd_config,
                 feature_dim,
                 num_classes,
                 backbone_type="resnet50",
                 pretrained=True):
        super(Model, self).__init__()
        self.backbone = Backbone(type=backbone_type, pretrained=pretrained)

        n = len(gd_config)
        k = feature_dim // n
        assert feature_dim % n == 0, 'The feature dim should be divided by the number of global desciptors'

        self.global_descriptors, self.main_modules = [], []
        for i in range(n):
            if gd_config[i] == 'S':
                p = 1
            elif gd_config[i] == 'M':
                p = float('inf')
            else:
                p = 3
            self.global_descriptors.append(GlobalDescriptor(p=p))
            self.main_modules.append(nn.Sequential(nn.Linear(2048, k, bias=False), L2Norm()))

        self.global_descriptors = nn.ModuleList(self.global_descriptors)
        self.main_modules = nn.ModuleList(self.main_modules)

        # Aux module
        self.auxiliary_module = nn.Sequential(nn.BatchNorm1d(2048), nn.Linear(2048, num_classes, bias=True))

    def forward(self, x):
        shared = self.backbone.backbone_modules(x)
        global_descriptors = []
        for i in range(len(self.global_descriptors)):
            global_descriptor = self.global_descriptors[i](shared)
            if i == 0:
                classes = self.auxiliary_module(global_descriptor)
            global_descriptor = self.main_modules[i](global_descriptor)
            global_descriptors.append(global_descriptor)
        global_descriptors = F.normalize(torch.cat(global_descriptors, dim=-1), dim=-1)
        return global_descriptors
