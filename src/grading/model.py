import torch
import torch.nn as nn
from torchvision import models
try:
    from src import config
except ImportError:
    import config

class EfficientNetB0Model(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, pretrained=True):
        super(EfficientNetB0Model, self).__init__()
        
        # Load EfficientNet-B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.model = models.efficientnet_b0(weights=weights)
        else:
            self.model = models.efficientnet_b0(weights=None)
            
        # Replace the classifier head
        # EfficientNet-B0 classifier input features: 1280
        num_features = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE if hasattr(config, 'DROPOUT_RATE') else 0.3),
            nn.Linear(num_features, 512),
            nn.SiLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def get_model(num_classes=config.NUM_CLASSES, pretrained=True):
    model = EfficientNetB0Model(num_classes=num_classes, pretrained=pretrained)
    return model
