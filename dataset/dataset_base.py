from torch.utils.data import Dataset
import numpy as np
import os

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()