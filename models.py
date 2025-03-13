
import torch
import torch.nn as nn
import torch.nn.functional as F


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_channels, out_channels, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.relu2 = nn.ReLU()

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out += self.shortcut(residual)
#         out = self.relu2(out)
#         return out

# class SignalPredictor(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(SignalPredictor, self).__init__()
#         self.in_channels = 64

#         self.num_classes = num_classes

#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7,4), padding=(3,0)) #padding = ?
#         self.sig1 = nn.Sigmoid()

#         self.layer1 = self._make_layer(BasicBlock, 64, 1, stride=1)
#         self.layer2 = self._make_layer(BasicBlock, 128, 1, stride=2)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(128 * BasicBlock.expansion, num_classes)

#     def _make_layer(self, block, out_channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = x.unsqueeze(1)

#         x = self.conv1(x)
#         x = self.sig1(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

#     def pre_fc(self, x):
#         x = x.unsqueeze(1)

#         x = self.conv1(x)
#         x = self.sig1(x)

#         x = self.layer1(x)
#         x = self.layer2(x)

#         x = self.avgpool(x)
#         return x

#     def swap_output_n(self, n):
#       self.fc = nn.Linear(self.fc.in_features, n)



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
        self.in_channels = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, stride=1, padding=3) # Adjust padding accordingly
        self.sig1 = nn.Sigmoid()

        self.layer1 = self._make_layer(BasicBlock1D, 64, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, 128, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128 * BasicBlock1D.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.sig1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def pre_fc(self, x):
        x = self.conv1(x)
        x = self.sig1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        return x

    def swap_output_n(self, n):
        self.fc = nn.Linear(self.fc.in_features, n)