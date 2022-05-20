import torch
torch.manual_seed(1741)
import random
random.seed(1741)
import numpy as np
np.random.seed(1741)
from torch.utils.data import Dataset

class EventDataset(Dataset):
    def __init__(self, data_instance) -> None:
        self.data = data_instance

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
