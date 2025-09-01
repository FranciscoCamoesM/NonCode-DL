
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


class SignalPredictor1D_Historical(nn.Module):  # THis is the architecture used in all models trained before 06/2025
    def __init__(self, num_classes=1000):
        super(SignalPredictor1D_Historical, self).__init__()
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


class SignalPredictor_Abstract(nn.Module):
    def __init__(self, num_datasets):
        super(SignalPredictor_Abstract, self).__init__()
        self.in_channels = 64

        self.flatten_dim = 128
        self.num_classes = 1
        self.num_datasets = num_datasets

        self.abstract = False

        self.conv1 = nn.Conv1d(4, 64, kernel_size=7, stride=1, padding=3) # Adjust padding accordingly
        self.relu1 = nn.ReLU()

        self.layer1 = self._make_layer(BasicBlock1D, 64, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D, self.flatten_dim, 1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_dim * BasicBlock1D.expansion, 1)

        self.deabstract = nn.Linear(1, num_datasets)

        self.sig = nn.Sigmoid()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, input_heads=None):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)

        if self.abstract:
            return x

        x = self.deabstract(x)
        
        if input_heads is None:
            raise ValueError("input_heads must be provided when abstract is False.")

        # Select specific outputs based on input_heads
        if len(input_heads) > x.shape[0]:
            input_heads = input_heads[:x.shape[0]]
        selected = torch.stack([x[i, head] for i, head in enumerate(input_heads)]).unsqueeze(1)
        
        return selected
    

class ComplexModel(torch.nn.Module):    # this model was meant to be used in regression tasks, but we ended up using the SignalPredictor1D model instead.
    def __init__(self, num_classes=1):
        super(ComplexModel, self).__init__()
        self.conv1 = nn.Conv1d(4, 128, kernel_size=7, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(128, 128, kernel_size=7, padding=3)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        self.conv5 = nn.Conv1d(128, 128*2, kernel_size=15, padding=0)
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.conv6 = nn.Conv1d(128*2, 128*4, kernel_size=25, padding=0)
        self.relu6 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.pool3 = nn.MaxPool1d(2)

        # self.conv7 = nn.Conv1d(128 * 4, 128 * 8, kernel_size=5, padding=1)
        # self.bn7 = nn.BatchNorm1d(128 * 8)
        # self.relu7 = nn.ReLU()
        # self.conv8 = nn.Conv1d(128 * 8, 128 * 8, kernel_size=5, padding=1)
        # self.bn8 = nn.BatchNorm1d(128 * 8)
        # self.relu8 = nn.ReLU()
        # self.pool4 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(128 * 24 * 2, 128*4)
        self.relufc1 = nn.ReLU()
        self.fc2 = nn.Linear(128*4, 128)
        self.relufc2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

        # self.normalization = nn.Linear(1, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.dropout1(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        x = self.pool3(x)

        # x = self.conv7(x)
        # x = self.bn7(x)
        # x = self.relu7(x)
        # x = self.conv8(x)
        # x = self.bn8(x)
        # x = self.relu8(x)
        # x = self.pool4(x)

        # print(f"Shape after conv layers: {x.shape}")

        x = x.view(-1, 128 * 24 * 2)
        x = self.fc1(x)
        x = self.relufc1(x)
        x = self.fc2(x)
        x = self.relufc2(x)
        x = self.fc3(x)
        # x = self.normalization(x)

        return x
    
    def abstract(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.dropout1(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        x = self.pool3(x)

        # x = self.conv7(x)
        # x = self.bn7(x)
        # x = self.relu7(x)
        # x = self.conv8(x)
        # x = self.bn8(x)
        # x = self.relu8(x)
        # x = self.pool4(x)

        # print(f"Shape after conv layers: {x.shape}")

        x = x.view(-1, 128 * 24 * 2)
        x = self.fc1(x)
        x = self.relufc1(x)
        x = self.fc2(x)
        x = self.relufc2(x)
        x = self.fc3(x)
        x = torch.mean(x, dim=1, keepdim=True)  # Average over the batch dimension

        return x