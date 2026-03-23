import timm
import torch.nn as nn


class LandmarkModel(nn.Module):
  def __init__(self, backbone_name, num_landmarks=68, pretrained=True):
    super().__init__()
    self.backbone = timm.create_model(
        backbone_name, 
        pretrained=pretrained, 
        num_classes=0
        )
    in_features = self.backbone.num_features
    self.head = nn.Linear(in_features, num_landmarks * 2)

  def forward(self, x):
    features = self.backbone(x)
    landmarks = self.head(features)
    return landmarks