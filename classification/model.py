import torch
import torch.nn as nn


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=16)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=16 * 874, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
