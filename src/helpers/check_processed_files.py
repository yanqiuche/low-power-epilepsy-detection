import os
from pyedflib import highlevel
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import signal
from src.helpers.correlations import cross_correlation, pearson_correlation, time_lagged_cross_correlation, window_time_lagged_cross_correlation
from pathlib import Path
from sklearn import preprocessing
import mne
from mne import read_proj
from mne.io import read_raw_edf
from time import time
from mne.preprocessing import ICA
from mne.datasets import sample

def import_files(folders=None, root=None):
    data = []
    if root is None:
        root = os.getcwd() + "/data/processed/"

    if folders is None:
        folders = ["normal/*.edf", "seizures/*.edf"]

    for i, folder in enumerate(folders):
        temp = ([(f, i) for f in sorted(Path(root).glob(folder))])
        data = data + temp

    return data


def detect_energy_spikes():
    seizure_file_1 = "/run/media/jmsvanrijn/3707BCE92020A60C/Data_2010_take_2/1.0.0/chb23/chb23_09.edf"
    normal_file = "/run/media/jmsvanrijn/3707BCE92020A60C/Data_2010_take_2/1.0.0/chb23/chb23_10.edf"
    start_time = 2000 # In seconds
    end_time = 3000 # In seconds
    hz = 256
    signals, signal_headers, header = highlevel.read_edf(str(seizure_file_1))

    recorded_seizures = np.array([0, 2589, 2660, 6885, 6947, 8505, 8532, 9580, 9664, len(signals[0])/hz])*hz
    seiz_23_1 = [29, 47]
    seiz_23_2 = [[30, 50], [53, 59]]
    seiz_23_3 = [2, 90]

    y_axis = np.resize([750, 0], len(recorded_seizures))

    z = np.array(signals[2])
    g = z[1:]**2 - z[1:]*z[:-1]
    g_2 = np.convolve(g, [1, 1, 1, 1, 1, 1, 1, 1])

    plt.subplot(211)
    plt.plot(np.transpose(z))
    plt.plot(recorded_seizures, y_axis, drawstyle="steps")

    y_axis = np.resize([np.max(g), 0], len(recorded_seizures))
    plt.subplot(212)
    plt.plot(np.transpose(g))
    plt.plot(np.transpose(g_2))
    plt.plot(recorded_seizures, y_axis, drawstyle="steps")
    plt.show()


def frequency_time_image():
    path = "/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/seizures/chb06_04_UY-seizure.edf"
    signal_norm, signal_header, _ = highlevel.read_edf(path)
    length = 30

    print(signal_header)
    f, t, Zxx = signal.stft(signal_norm[2], nperseg=8, fs=256)

    print(np.abs(Zxx).tostring)

    plt.pcolormesh(t, f, np.abs(Zxx))
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.show()


def normalize_data(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled = min_max_scaler.fit_transform(data)

    # result = (data - np.min(data) / (np.max(data) - np.min(data)))
    return scaled

    select_patients = [1, 22]

    path = "/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/"

    indexed_seizures = [([str(i), str(i).split("/")[-1].split("_")[-3], str(i).split("/")[-1].split("_")[-2]]) for i in sorted(Path(path).glob('seizures/*.edf'))]
    indexed_normal = [([str(i), str(i).split("/")[-1].split("_")[-3], str(i).split("/")[-1].split("_")[-2]]) for i in sorted(Path(path).glob('normal/*.edf'))]

    # fft_per_second(indexed_seizures[30])
    # y_corr, x_corr = pearson_correlation(indexed_normal, indexed_seizures)
    # time_lagged_cross_correlation(indexed_seizures[10], x_corr, y_corr)
    # cross_correlation(x_corr, y_corr, indexed_seizures[10])

    plt.show()

def fft_per_second(data):
    signal_, _, _ = highlevel.read_edf(data[0])
    hz = 256
    end_time = 2

    # fft = np.fft.fft(signal_[1, 0:hz*end_time], norm="ortho")
    # fftfreq = np.fft.fftfreq(len(signal_[1, :hz*end_time]))

    f, t, Zxx = signal.stft(signal_[18][:256*5], nperseg=64, fs=256)
    print((signal_)[:, 0:256*5].shape)
    f_, t_, Zxx_ = signal.stft(siguitargnal_[:, :256*5], nperseg=64, fs=256)
    print(Zxx.shape)
    print(Zxx_.shape)

    plt.figure()
    plt.pcolor(t, f, normalize_data(np.abs(Zxx)), cmap="RdBu")
    plt.colorbar()
    # plt.grid()

    plt.figure()
    plt.pcolor(t_, f_, normalize_data(np.abs(Zxx_[18])), cmap="RdBu")
    # plt.semilogy(fftfreq[:256]*256, np.abs(fft[:256]))
    # plt.grid()
    plt.show()

def run_ica(method, picks, reject, raw, fit_params=None):
    ica = ICA(n_components=20, method=method, fit_params=fit_params,
              random_state=0)
    t0 = time()
    print(raw)
    ica.fit(raw, picks=picks, reject=reject)
    fit_time = time() - t0
    title = ('ICA decomposition using %s (took %.1fs)' % (method, fit_time))
    ica.plot_components(title=title)


def projection_on_head(file):
    print(file[0])
    raw = read_raw_edf(file[0], preload=True)
    raw.pick_types(eeg=True, stim=True)

    picks = mne.pick_types(raw.info)
    print("Print picks: ")
    print(picks)
    reject = dict(mag=5e-12, grad=4000e-13)
    raw.filter(1, 30, fir_design='firwin')
    run_ica('fastica', picks, reject, raw)



def main_thing():
    select_patients = [1, 22]
    path = "/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/"

    indexed_seizures = [([str(i), str(i).split("/")[-1].split("_")[-3], str(i).split("/")[-1].split("_")[-2]]) for i in sorted(Path(path).glob('seizures/*.edf'))]
    indexed_normal = [([str(i), str(i).split("/")[-1].split("_")[-3], str(i).split("/")[-1].split("_")[-2]]) for i in sorted(Path(path).glob('normal/*.edf'))]

    projection_on_head(indexed_seizures[30])
    # fft_per_second(indexed_seizures[30])
    # y_corr, x_corr = pearson_correlation(indexed_normal, indexed_seizures)
    # time_lagged_cross_correlation(indexed_seizures[10], x_corr, y_corr)
    # cross_correlation(x_corr, y_corr, indexed_seizures[10])
    # window_time_lagged_cross_correlation(x_corr, y_corr)

    plt.show()


main_thing()
