import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov5 import YOLOv5

#region feature extraction
class FeatureExtractor(nn.Module):
    def __init__(self, yolov5):
        super(FeatureExtractor, self).__init__()
        # Load the pre-trained YOLOv5 model
        self.feature_extractor = nn.Sequential(*list(yolov5.model.children())[:-1])  # All layers except the detection head

    def forward(self, x):
        return self.feature_extractor(x)
#endregion
#region Detection layer    

class DetectionHead(nn.Module):
    def __init__(self, yolov5):
        super(DetectionHead, self).__init__()
        # Load the pre-trained YOLOv5 model
        self.detection_head = yolov5.model[-1]  # Only the detection head    
    def forward(self, x):
        return self.detection_head(x)
#endregion