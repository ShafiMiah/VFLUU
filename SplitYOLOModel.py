import torch
import torch.nn as nn
from ultralytics import YOLO

class FeatureExtractor(nn.Module):
    def __init__(self, yolov8_model):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = nn.Sequential(*list(yolov8_model.model.children())[:-1])

    def forward(self, x):
        return self.feature_extractor(x)

class DetectionHead(nn.Module):
    def __init__(self, yolov8_model):
        super(DetectionHead, self).__init__()
        self.detection_head = yolov8_model.model[-1]

    def forward(self, x):
        return self.detection_head(x)