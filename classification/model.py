import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_dims=1, dropout_prob=0.3):
        super(CNNModel, self).__init__()

        # Dropout probability
        self.dropout_prob = dropout_prob

        # First 1D convolutional layer (4 filters, kernel size of 16)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=16, stride=1, padding="same")

        # Batch normalization for the first convolutional layer
        self.bn1 = nn.BatchNorm1d(4)

        # Second 1D convolutional layer (8 filters, kernel size of 8)
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=8, kernel_size=8, stride=1, padding="same")

        # Batch normalization for the second convolutional layer
        self.bn2 = nn.BatchNorm1d(8)

        # Max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=8, stride=4)

        # Dropout layer to avoid overfitting
        self.dropout = nn.Dropout(p=self.dropout_prob)

        # Fully connected layer
        self.fc1 = nn.Linear(input_dims * 8 // 4, 128)

        # Output layer for regression
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Reshape the input to match Conv1D's expected shape (batch_size, channels, sequence_length)
        x = x.unsqueeze(1)

        # First Conv1D layer followed by BatchNorm, GELU, and Dropout
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Second Conv1D layer followed by BatchNorm, GELU, and Dropout
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        # Max pooling layer
        x = self.pool(x)

        # Flatten the tensor for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layer followed by ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer (linear activation)
        x = self.fc2(x)

        return x
