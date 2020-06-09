from pathlib import Path
from pyedflib import highlevel
import math
import random
from torch.utils.data import DataLoader
from src.models.epilepsy_data_loader import MultiplePerFileEpilepsyData, OnePerFileEpilepsyData


def load_data(path, window_size, bs, type, train_ratio):
    if type == 1:
        all_files = one_per_file(path)
        train_data = OnePerFileEpilepsyData(all_files[:round(train_ratio * len(all_files))], window_size)
        valid_data = OnePerFileEpilepsyData(all_files[round(train_ratio * len(all_files)):], window_size)
    else:
        all_files = max_per_file(path, window_size)
        train_data = MultiplePerFileEpilepsyData(all_files[:round(train_ratio * len(all_files))], window_size)
        valid_data = MultiplePerFileEpilepsyData(all_files[round(train_ratio * len(all_files)):], window_size)

    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=bs, shuffle=True)
    return train_loader, valid_loader


def one_per_file(path):
    seizure_files = [(str(seizure), str(1)) for seizure in sorted(Path(path).glob('seizures/*.edf'))] # 1 = seizure
    normal_files = [(str(normal), str(0)) for normal in sorted(Path(path).glob('normal/*.edf'))]
    return random.sample(seizure_files + normal_files, len(seizure_files+normal_files))


def max_per_file(path, window_size):
    seizure_files = []
    normal_files = []
    for seizure in sorted(Path(path).glob('seizures/*.edf')):
        signals, _, _ = highlevel.read_edf(str(seizure))
        length_signals = int(math.floor(len(signals[0]) / window_size))
        for i in range(length_signals):
            seizure_files.append((str(seizure), str(1), str(i)))

    for normal in sorted(Path(path).glob('normal/*.edf')):
        signals, _, _ = highlevel.read_edf(str(normal))
        length_signals = int(math.floor(len(signals[0]) / window_size))
        for i in range(length_signals):
            normal_files.append((str(normal), str(0), str(i)))

    return random.sample(seizure_files + normal_files, len(seizure_files+normal_files))
