import torch
import torch.nn as nn

class DeepTreeModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepTreeModule, self).__init__()
        # Nhánh 1: Sử dụng kernel size 3 (tập trung vào texture)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Nhánh 2: Sử dụng kernel size 5 (tập trung vào hình dạng)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # Hợp nhất đầu ra của các nhánh
        self.fc = nn.Linear(out_channels * 2, out_channels)

    def forward(self, x):
        # x: [B, C, H, W]
        b1 = self.branch1(x)  # [B, out_channels, 1, 1]
        b2 = self.branch2(x)  # [B, out_channels, 1, 1]
        # Flatten các đầu ra
        b1 = b1.view(x.size(0), -1)
        b2 = b2.view(x.size(0), -1)
        # Nối chập theo chiều kênh
        combined = torch.cat([b1, b2], dim=1)
        out = self.fc(combined)
        return out