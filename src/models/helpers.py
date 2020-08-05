from pathlib import Path
from pyedflib import highlevel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def load_correlations(select_patients):
    path = "/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/"
    index_seizures = [([str(i), str(i).split("/")[-1].split("_")[-3], str(i).split("/")[-1].split("_")[-2]]) for i in
                        sorted(Path(path).glob('seizures/*.edf'))]
    index_normal = [([str(i), str(i).split("/")[-1].split("_")[-3], str(i).split("/")[-1].split("_")[-2]]) for i in
                      sorted(Path(path).glob('normal/*.edf'))]
    seizures = []
    normal = []
    selected_seizures = []
    selected_normal = []

    for i, seizure in enumerate(index_seizures):
        try:
            patient_numbers = int(seizure[1][-2:])
        except:
            patient_numbers = int(seizure[1][-3:-1])

        if patient_numbers in select_patients:
            selected_seizures.append(seizure)
            selected_normal.append(index_normal[i])

    print("AoSeizures: " + str(len(selected_seizures)))
    print("AoNormals: " + str(len(selected_normal)))

    for i in range(len(selected_normal)):
        signal_seiz,  _, _ = highlevel.read_edf(str(selected_seizures[i][0]))
        normal_seiz, _, _ = highlevel.read_edf(str(selected_normal[i][0]))
        seizures.append((signal_seiz, str(1)))
        normal.append((normal_seiz, str(0)))

    return seizures, normal


def calculate_window(data, window, time):
    frame = pd.DataFrame(np.zeros((23, int(window)*256)))
    i = 0
    for seizures in data:
        t_start = ((window-1) + time) * 256
        t_end = (window + time) * 256

        try:
            selected = seizures[:, t_start:t_end]
        except:
            selected = seizures[0][:, t_start:t_end]


        if len(selected[1]) != t_end-t_start:
            i = i + 1
        else:
            frame = frame.add(pd.DataFrame(np.abs(selected)))
            # frame = frame.add(pd.DataFrame(selected))

    return frame.T.corr(), i


def calculated_patients():
    window = 10
    time = 10
    patients = np.arange(0, 22)
    nums_bins = 20

    seizures, normal = load_correlations(patients)
    hist_seiz = []
    hist_nor = []

    for w in range(1, window): # Window start at 1 as 0 will lead to an empty window
        for t in range(0, time):
            nor_corr, nor_drop = calculate_window(seizures, w, t)
            seiz_corr, seiz_drop = calculate_window(normal, w, t)
            # print(seiz_corr)
            seiz_corr.to_numpy()[np.diag_indices(23)] = 0
            print(pd.DataFrame(seiz_corr))
            print("Size seiz: " + str(round(seiz_corr.to_numpy().sum(), 2)) + " Size nor: " + str(round(
                nor_corr.to_numpy().sum(), 2)) + " Win: " + str(w) + " Time: " + str(t), " Dropped: " + str(nor_drop))
            hist_seiz.append(seiz_corr.to_numpy().sum()/(2*(len(seizures)-seiz_drop)))
            hist_nor.append(nor_corr.to_numpy().sum()/(2*(len(normal)-nor_drop)))

    print(hist_seiz)
    fig, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].set_title("Seizure distribution")
    axes[0].hist(hist_seiz, nums_bins)
    axes[1].set_title("Normal distribution")
    axes[1].hist(hist_nor, nums_bins)

    # plt.show()

# calculated_patients()
