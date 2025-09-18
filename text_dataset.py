import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_ids, context_length: int, pad_id: int):
        self.data = torch.tensor(data_ids, dtype=torch.long)
        self.context_length = context_length
        self.pad_id = pad_id
    def __len__(self):
        return max(0, len(self.data) - self.context_length - 1)
    def __getitem__(self, idx):
        x = self.data[idx: idx + self.context_length]
        y = self.data[idx + 1: idx + self.context_length + 1]
        return x, y