from torch.utils.data import Dataset
import torch

class SpectralDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # the data is pd.DataFrame
        data = self.data.iloc[idx].values
        label = self.labels.iloc[idx].values
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)