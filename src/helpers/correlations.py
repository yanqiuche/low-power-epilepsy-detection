import pandas as pd
import numpy as np
from pyedflib import highlevel
from matplotlib import pyplot as plt
import math
from scipy import signal
from matplotlib.widgets import Slider, Button, RadioButtons


def cross_correlation_widget():
    fig, ax = plt.subplot()
    axwindow = plt.axes([0.25, 0.1, 0.65, 0.03])
    window = Slider(axwindow ,5, 10, valint=10)


def cross_correlation(datax, datay, path):
    seconds = 20
    fps = 256
    signal, _, _ = highlevel.read_edf(path[0])

    d1 = pd.Series(signal[datax])
    d2 = pd.Series(signal[datay])
    rs = [(d1.corr(d2.shift(lag))) for lag in range(-int(seconds * fps), int(seconds * fps + 1))]

    offset = np.ceil(len(rs) / 2) - np.argmax(rs)
    f, ax = plt.subplots(figsize=(14, 3))
    ax.plot(rs)
    ax.axvline(np.ceil(len(rs) / 2), color='k', linestyle='--', label='Center')
    ax.axvline(np.argmax(rs), color='r', linestyle='--', label='Peak synchrony')
    ax.set(title=f'Offset = {offset} frames\nS1 leads <> S2 leads', xlabel='Offset',
           ylabel='Pearson r')
    ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])
    plt.legend()


def pearson_correlation(indexed_normal, index_seizures):
    frame_normal = pd.DataFrame(np.zeros((23, 23)))
    frame_seiz = pd.DataFrame(np.zeros((23, 23)))

    for i in range(len(index_seizures)-150):
        signal_nor, _, _ = highlevel.read_edf(str(indexed_normal[i][0]))
        signal_seiz,  _, _ = highlevel.read_edf(str(index_seizures[i][0]))

        frame_normal = frame_normal.add(pd.DataFrame(signal_nor[:15*256]))
        frame_seiz = frame_seiz.add(pd.DataFrame(signal_seiz[:15 * 256]))

        # frame_normal = frame_normal.add(pd.DataFrame(abs(signal_nor)).T.corr())
        # frame_seiz = frame_seiz.add(pd.DataFrame(abs(signal_seiz)).T.corr())

    normal_corr = frame_normal.T.corr()
    seiz_corr = frame_seiz.T.corr()

    size = 23
    high_correlation = frame_seiz.to_numpy()
    lowest = np.argmin(high_correlation.reshape(-1), axis=0)
    y_corr = math.floor(lowest/size)
    x_corr = lowest % size

    # plt.figure()
    # plt.imshow(normal_corr, origin="lower", cmap="hot", interpolation="nearest")
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(seiz_corr, origin="lower", cmap="hot", interpolation="nearest")
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(np.abs(normal_corr), origin="lower", cmap="hot", interpolation="nearest")
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(np.abs(seiz_corr), origin="lower", cmap="hot", interpolation="nearest")
    # plt.colorbar()
    #
    # plt.figure()
    # plt.imshow(np.abs(seiz_corr)-np.abs(normal_corr), origin="lower", cmap="hot", interpolation="nearest")
    # plt.colorbar()

    # return x_corr, y_corr

    return normal_corr

def time_lagged_cross_correlation(path, sig_1=2, sig_2=6):
    signal_norm, signal_header, _ = highlevel.read_edf(path[0])

    # Load data
    signal_1 = pd.DataFrame(signal_norm[sig_1])
    signal_2 = pd.DataFrame(signal_norm[sig_2])

    # Set window size to compute moving window synchrony.
    r_window_size = 120

    # Compute rolling window synchrony
    rolling_r = signal_1.rolling(window=r_window_size, center=True).corr(signal_2)
    f, ax = plt.subplots(2, 1, figsize=(14, 6))
    signal_1.rolling(window=30, center=True).median().plot(ax=ax[0])
    signal_2.rolling(window=30, center=True).median().plot(ax=ax[0])
    ax[0].set(xlabel='Frame', ylabel='Original signals')
    rolling_r.plot(ax=ax[1])
    ax[1].set(xlabel='Frame', ylabel='Pearson R')
    plt.suptitle("Smiling data and rolling window correlation")


def window_time_lagged_cross_correlation(x, y):
    # Windowed time lagged cross correlation
    path = "/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/seizures/chb06_04_UY-seizure.edf"
    signal, _, _ = highlevel.read_edf(path)

    series_1 = pd.Series(signal[x])
    series_2 = pd.Series(signal[y])

    seconds = 5
    fps = 30
    no_splits = 20
    samples_per_split = len(signal[x]) / no_splits
    rss = []
    for t in range(0, no_splits):
        d1 = series_1.loc[(t) * samples_per_split:(t + 1) * samples_per_split]
        d2 = series_2.loc[(t) * samples_per_split:(t + 1) * samples_per_split]
        # rs = [crosscorr(d1, d2, lag) for lag in range(-int(seconds * fps), int(seconds * fps + 1))]
        rs = [(d1.corr(d2.shift(lag))) for lag in range(-int(seconds * fps), int(seconds * fps + 1))]
        rss.append(rs)

    rss = pd.DataFrame(rss)
    f, ax = plt.subplots(figsize=(10, 5))
    print(rss)
    # sns.heatmap(rss, cmap='RdBu_r', ax=ax)
    plt.pcolor(rss, cmap="RdBu")

    ax.set(title=f'Windowed Time Lagged Cross Correlation', xlim=[0, 301], xlabel='Offset', ylabel='Window epochs')
    ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])

    # Rolling window time lagged cross correlation
    seconds = 5
    fps = 30
    window_size = 300  # samples
    t_start = 0
    t_end = t_start + window_size
    step_size = 30
    rss = []
    while t_end < 5400:
        d1 = series_1.iloc[t_start:t_end]
        d2 = series_2.iloc[t_start:t_end]
        # rs = [crosscorr(d1, d2, lag, wrap=False) for lag in range(-int(seconds * fps), int(seconds * fps + 1))]
        rs = [(d1.corr(d2.shift(lag))) for lag in range(-int(seconds * fps), int(seconds * fps + 1))]
        rss.append(rs)
        t_start = t_start + step_size
        t_end = t_end + step_size
    rss = pd.DataFrame(rss)

    f, ax = plt.subplots(figsize=(10, 10))
    plt.pcolor(rss, cmap="RdBu")
    ax.set(title=f'Rolling Windowed Time Lagged Cross Correlation', xlim=[0, 301], xlabel='Offset', ylabel='Epochs')
    ax.set_xticks([0, 50, 100, 151, 201, 251, 301])
    ax.set_xticklabels([-150, -100, -50, 0, 50, 100, 150])