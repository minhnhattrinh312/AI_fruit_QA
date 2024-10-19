import torch
import torch.nn as nn


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16, stride=1, padding="same")
        self.batchnorm1 = nn.BatchNorm1d(num_features=16)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, stride=1, padding="same")
        self.batchnorm2 = nn.BatchNorm1d(num_features=32)
        self.gelu2 = nn.GELU()

        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding="same")
        self.batchnorm3 = nn.BatchNorm1d(num_features=64)
        self.gelu3 = nn.GELU()
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same")
        self.batchnorm4 = nn.BatchNorm1d(num_features=128)
        self.gelu4 = nn.GELU()

        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=128 * 218, out_features=128)  # Adjusted in_features
        self.gelu5 = nn.GELU()
        self.dense2 = nn.Linear(in_features=128, out_features=64)
        self.gelu6 = nn.GELU()
        self.dropout3 = nn.Dropout(p=0.2)
        self.dense3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.gelu2(x)

        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.gelu3(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.gelu4(x)

        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.gelu5(x)
        x = self.dense2(x)
        x = self.gelu6(x)
        x = self.dropout3(x)
        x = self.dense3(x)
        return x


# Example usage
if __name__ == "__main__":
    model = BasicModel()
    input_tensor = torch.randn(32, 1, 874)  # batch_size=32, channels=1, sequence_length=874
    output = model(input_tensor)
    print(output.shape)
