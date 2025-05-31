import torch
from torch.utils.data import Dataset
import numpy as np


class torch_Dataset(Dataset):
    def __init__(self, data):
        
        sequences = [seq[None, ...] for seq in data.sequence]  # (1, T, H, W)
        sequences = np.stack(sequences)  # (N, 1, T, H, W)

      
        self.data = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(data.label.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]