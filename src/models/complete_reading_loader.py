from torch.utils.data import Dataset
from pyedflib import highlevel
import pandas as pd
import numpy as np


class CompleteReading(Dataset):
    def __init__(self, file_name, sample_spacing, window_size):
        signals, signal_headers, header = highlevel.read_edf(file_name)

        self.sample_spacing = sample_spacing
        self.window_size = window_size
        self.signals = signals
        self.items = range(int(len(signals[0])/sample_spacing - window_size/sample_spacing - 1))
        self.length = len(self.items)

    def __getitem__(self, index):
        start = self.items[index]
        return self.signals[:, start*self.sample_spacing:(start+4)*self.sample_spacing]

    def __len__(self):
        return self.length


class CompleteReading1D_CNN(Dataset):
    def __init__(self, files):
        window = 2
        start = 2
        # self.signals = [(self.create_corr_matrix(signal, window, start)) for signal in files]
        # self.signals = self.create_corr_matrix_all(files, window)
        self.signals = self.create_corr_matrix_2(files, window)
        self.length = len(self.signals)

    def __getitem__(self, item):
        return self.signals[item]

    def __len__(self):
        return self.length

    def create_corr_matrix(self, signal, window, start):
        t_start = ((window - 1) + start) * 256
        t_end = (window + start) * 256
        selected = signal[0][:, t_start:t_end]
        frame = pd.DataFrame(selected).T.corr().to_numpy()
        return frame,  int(signal[1])

    def create_corr_matrix_all(self, files, window):
        signal = []

        for file in files:
            t_times = len(file[0][0]) // (window*256)
            for t in range(t_times):
                t_start = ((window - 1) + t) * 256
                t_end = (window + t) * 256
                selected = file[0][:, t_start:t_end]
                frame = pd.DataFrame(selected).T.corr().to_numpy()
                signal.append((frame, int(file[1])))

        return signal

    def create_corr_matrix_2(self, files, window):
        signal = []

        for file in files:
            t_times = len(file[0][0]) // (window * 256)
            for t in range(t_times-1):
                t_start = ((window - 1) + t) * 256
                t_end = (window + t) * 256
                selected_1 = file[0][:, t_start+(window*256):t_end+(window*256)]
                selected_2 = file[0][:, t_start:t_end]
                frame_1 = pd.DataFrame(selected_1).T.corr().to_numpy()
                frame_2 = pd.DataFrame(selected_2).T.corr().to_numpy()
                signal.append((np.subtract(frame_1, frame_2), int(file[1])))

        return signal


