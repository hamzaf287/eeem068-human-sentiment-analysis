import torch
from torchvision import models

def get_full_image_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    return model