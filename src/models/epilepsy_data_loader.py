from torch.utils.data import Dataset
from pyedflib import highlevel
import random


class OnePerFileEpilepsyData(Dataset):
    def __init__(self, data, window_size):
        # Get directory listing from path
        self.window_size = window_size
        self.items = data

        self.length = len(self.items)

    def __getitem__(self, index):
        filename, label = self.items[index]
        # t = time.time()
        signals, signal_headers, header = highlevel.read_edf(filename)
        # print(time.time()-t)
        loc = random.randint(0, len(signals[0]) - self.window_size)
        signals_cut = signals[:, loc: loc + self.window_size]
        return signals_cut, int(label), filename

    def __len__(self):
        return self.length


class MultiplePerFileEpilepsyData(Dataset):
    def __init__(self, data, window_size):
        # Get directory listing from path
        self.window_size = window_size
        self.items = data
        self.length = len(self.items)

    def __getitem__(self, index):
        
        filename, label, start = self.items[index]
        # t = time.time()
        signals, signal_headers, header = highlevel.read_edf(filename)
        # print(time.time()-t)

        signal_cut = signals[:, int(start)*self.window_size: (int(start)+1)*self.window_size]
        return signal_cut, int(label)

    def __len__(self):
        return self.length


class MultiplePerFileEpilepsyData2(Dataset):
    def __init__(self, data):
        # Get directory listing from path
        self.items = data
        self.length = len(self.items)

    def __getitem__(self, index):
        recording, label = self.items[index]
        return recording, int(label)

    def __len__(self):
        return self.length
