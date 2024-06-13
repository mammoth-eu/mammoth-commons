import os
from mammoth.models.pytorch import Pytorch
from mammoth.integration import loader
from torchvision.models import resnet
import torch.nn as nn


@loader(namespace="gsarridis", version="v003", python="3.11")
def model_torch(path: str, model_dict: str = "", model_class="") -> Pytorch:
    """This is an PYTORCH loader."""
    # TODO get the model class from os.path.join(path,model_class) .py file
    model_class = resnet.resnet18()
    model_class.fc = nn.Linear(model_class.fc.in_features, 2)
    return Pytorch(os.path.join(path, model_dict), model_class)