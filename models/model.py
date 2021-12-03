import torch
import torch.nn as nn

from backbone import Backbone
from layers import GlobalDescriptor


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.backbone = Backbone()
