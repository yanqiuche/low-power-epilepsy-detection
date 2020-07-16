from torch.utils.data import Dataset
from pyedflib import highlevel
from scipy import signal
from src.models.helpers import load_correlations, calculate_window
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
        patients = np.arange(1, 10)
        window, time = 1024
        normal, seizures = load_correlations(patients)
        calculate_window()
        print(normal)
        self.length = len(files[0])

        print(len(self.signals[0]))
        print(len(self.signals[0][0]))
        print(len(self.signals[0][0][0]))

    def __getitem__(self, item):
        return self.signals

    def __len__(self):
        return self.length

    def create_stft(self, file):
        f, t, Zxx = np.abs(signal.stft(file[0], nperseg=64, fs=256))
        return [Zxx, file[1]]
