import timm
import torch.nn as nn
import torch


class LandmarkModel(nn.Module):
  def __init__(self, backbone_name, num_landmarks=68, pretrained=True):
    super().__init__()
    self.backbone = timm.create_model(
        backbone_name, 
        pretrained=pretrained, 
        num_classes=0,
        global_pool='avg'
        )
    
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 112, 112)
        in_features = self.backbone(dummy).shape[1]

    self.head = nn.Linear(in_features, num_landmarks * 2)

  def forward(self, x):
    features = self.backbone(x)
    landmarks = self.head(features)
    return landmarks