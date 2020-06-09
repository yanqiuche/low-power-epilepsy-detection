from matplotlib import pyplot as plt
from src.models.learn_rate import find_lr
from pathlib import Path
from pyedflib import highlevel
import numpy as np


def data_length(folder):
    seizure_files = [(str(seizure)) for seizure in sorted(Path(folder).glob('seizures/*.edf'))]
    normal_files = [(str(normal)) for normal in sorted(Path(folder).glob('normal/*.edf'))]
    print("Number of files with seizures: " + str(len(seizure_files)))
    print("Number of files without siezures: " + str(len(normal_files)))

    signals, _, _ = highlevel.read_edf(str(seizure_files[20]))
    print(signals[10])

    length_seizures = []
    length_normal = []
    for seizure_file in seizure_files:
        signals, _, _ = highlevel.read_edf(str(seizure_file))
        length_seizures.append(len(signals[0]))
    for normal_file in normal_files:
        signals, _, _ = highlevel.read_edf(str(normal_file))
        length_normal.append(len(signals[0]))

    dis = np.sum(np.array(length_seizures)) # data in seizure
    din = np.sum(np.array(length_normal)) # data in normal

    print("Seizure: " + str(dis) + "fr/ " + str(dis/256) + "s/ " + str(dis/256/3600) + "h")
    print("Normal: " + str(din) + "fr/ " + str(din/256) + "s/ " + str(din/256/3600) + "h")


# data_length("/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/")