
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)

        out = self.relu2(out)
        return out

class SignalPredictor1D(nn.Module):
    def __init__(self, num_classes=1000):
        super(SignalPredictor1D, self).__init__()

        L1 = 64
        B1 = 64
        B2 = 128

        self.in_channels = L1
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(4, L1, kernel_size=7, stride=1, padding=3) # Adjust padding accordingly
        self.relu1 = nn.ReLU()

        self.layer1 = self._make_layer(BasicBlock1D, B1, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, B2, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.1)  # Add dropout layer for regularization
        self.fc1 = nn.Linear(B2 * BasicBlock1D.expansion, 64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)  # Final classification output

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        

        return x

    def pre_fc(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        return x

    def swap_output_n(self, n):
        self.fc = nn.Linear(self.fc.in_features, n)

    def abstract(self, x):
        if self.num_classes == 1:
            return self(x)
        else:
            x = self(x)
            return torch.mean(x, dim=1, keepdim=True)