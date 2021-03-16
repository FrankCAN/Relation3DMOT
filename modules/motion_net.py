import torch
import torch.nn as nn


class MotionNet(nn.Module):

    def __init__(self, in_channels, out_channels=512):
        super(MotionNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.bn1 = nn.GroupNorm(64, 64)

        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.GroupNorm(64, 64)

        self.conv3 = nn.Conv1d(64, 128, 1)
        self.bn3 = nn.GroupNorm(128, 128)

        self.conv4 = nn.Conv1d(128, out_channels, 1)
        self.bn4 = nn.GroupNorm(out_channels, out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        out = x

        return out
