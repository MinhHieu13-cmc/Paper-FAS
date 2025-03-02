import torch
import torch.nn as nn
from torchvision import models
from LAB.DeepTreeLearning import DeepTreeModule
from LAB.APersonalizedBenchmark import PersonalizationModule

class FaceAntiSpoofingModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FaceAntiSpoofingModel, self).__init__()
        # Sử dụng MobileNetV3 làm backbone (pretrained)
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        # Lấy phần features của MobileNetV3 (loại bỏ classifier)
        self.backbone = mobilenet.features
        # Kích thước đầu ra của MobileNetV3_small thường là 576 channels
        backbone_out_channels = 576

        # Module Deep Tree Learning
        deep_tree_out_channels = 256  # có thể tùy chỉnh
        self.deep_tree = DeepTreeModule(in_channels=backbone_out_channels,
                                        out_channels=deep_tree_out_channels)

        # Module cá nhân hóa: từ đặc trưng deep tree đến không gian cá nhân hóa
        personalized_dim = 128
        self.personalization = PersonalizationModule(feature_dim=deep_tree_out_channels,
                                                     personalized_dim=personalized_dim)

        # Lớp phân loại cuối cùng
        self.classifier = nn.Linear(deep_tree_out_channels, num_classes)

    def forward(self, x):
        # x: [B, 3, H, W]
        features = self.backbone(x)   # [B, 576, H', W']
        tree_features = self.deep_tree(features)  # [B, 256]
        # Áp dụng module cá nhân hóa
        personalized_features = self.personalization(tree_features)  # [B, 256]
        logits = self.classifier(personalized_features)  # [B, num_classes]
        return logits


