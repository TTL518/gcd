import torch
import torch.nn as nn
import torchvision.models as models


class Backbone(nn.Module):
    def __init__(self,
                 type="resnet50",
                 pretrained=True):
        super(Backbone, self).__init__()
        self.type = type
        self.module = models.resnet50(pretrained=pretrained)
