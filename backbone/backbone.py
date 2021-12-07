import torch
import torch.nn as nn
import torchvision.models as models


class Backbone(nn.Module):
    def __init__(self,
                 type="resnet50",
                 pretrained=True):
        super(Backbone, self).__init__()
        self.type = type
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone_modules = self._get_backbone_params()

    def _get_backbone_params(self):
        params = []
        for name, module in self.backbone.named_children():
            if isinstance(module, nn.AdaptiveAvgPool2d) or isinstance(module, nn.Linear):
                continue
            params.append(module)
        return nn.Sequential(*params)
