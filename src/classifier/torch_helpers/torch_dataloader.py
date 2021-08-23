import pandas as pd
import torch
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import TensorDataset, DataLoader, Dataset
import multiprocessing as mp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RegardDataset(Dataset):
    def __init__(self, data, labels):
        super(RegardDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, i):
        return self.labels[i], self.data[i]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i, x in enumerate(self.data):
            yield self.labels[i], x

    def get_labels(self):
        return self.labels


def get_dataloader(X, Y, batch_size, shuffle=True):
    if isinstance(Y, pd.Series):
        Y = Y.values.astype("int")
    data = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=mp.cpu_count(),
    )
    return dataloader
