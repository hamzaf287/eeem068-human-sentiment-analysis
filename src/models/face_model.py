import torch.nn as nn
import torchvision.models as models

class FaceSentimentModel(nn.Module):
    def __init__(self, num_classes=3):
        super(FaceSentimentModel, self).__init__()
        # Using the modern torchvision weights parameter
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)