from torch.utils.data import Dataset
from pyedflib import highlevel
import random
import numpy as np


class EpilepsyData(Dataset):
    def __init__(self, data, window_size):
        # Get directory listing from path
        self.window_size = window_size
        self.items = data
        self.length = len(self.items)

    def __getitem__(self, index):
        filename, label = self.items[index]
        signals, signal_headers, header = highlevel.read_edf(filename)

        loc = random.randint(0, len(signals[0]) - self.window_size)
        signals_cut = signals[:, loc: loc + self.window_size]
        signal_mean = []
        # for i in range(int(len(signals_cut[0])/2)-1):
        #     t = 2*i
        #     signal_mean.append(signals_cut[:, t] + signals_cut[:, t+1])
        return signals_cut, int(label), filename

    def __len__(self):
        return self.length
