import torch
from torchvision import models


def get_full_image_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = torch.nn.Linear(model.fc.in_features, 3)

    return model
