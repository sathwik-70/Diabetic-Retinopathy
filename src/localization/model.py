import torch
import torch.nn as nn
from torchvision import models
try:
    from src import config
except ImportError:
    import config

class LocalizationModel(nn.Module):
    def __init__(self, pretrained=True):
        super(LocalizationModel, self).__init__()
        
        # Backbone: ResNet50
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            self.backbone = models.resnet50(weights=None)
            
        # Replace FC
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, 4), # 4 coordinates: OD_x, OD_y, Fovea_x, Fovea_y
            nn.Sigmoid() # Output 0-1
        )

    def forward(self, x):
        return self.backbone(x)

def get_model(pretrained=True):
    return LocalizationModel(pretrained=pretrained)
