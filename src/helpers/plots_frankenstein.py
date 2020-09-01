import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider
from src.models.helpers import load_correlations, calculate_window
import random
from scipy.stats import shapiro, norm
import scipy.stats as stats
import pandas as pd

def create_random_signal(seizures_data, normal_data, span_time):
    # random.seed(35)
    s_time = random.randint(0, int(((len(seizures_data[0][0][0]) / 256) - span_time)))
    seizures = np.array(seizures_data[0][0][:, s_time * 256:(s_time + span_time) * 256])
    normals = np.array(normal_data[0][0][:,  s_time * 256:(s_time + span_time) * 256])
    frankenstein_signal = np.hstack((seizures, normals))

    for i in range(1, len(seizures_data)):
        print(str(((len(seizures_data[i][0][0]) / 256))))
        s_time = random.randint(0, int(((len(seizures_data[i][0][0]) / 256) - span_time)))
        seizures = np.array(seizures_data[i][0][:,  s_time * 256:(s_time + span_time) * 256])
        normals = np.array(normal_data[i][0][:,  s_time * 256:(s_time + span_time) * 256])
        frankenstein_signal = np.hstack((frankenstein_signal, seizures))
        frankenstein_signal = np.hstack((frankenstein_signal, normals))

    return frankenstein_signal


def max_channel_power(frankenstein_signal, start, window):
    # frankenstein_signal = frankenstein_signal[:, (start-window)*256:(start + 2*window)*256]
    power_chans = []
    for channel in frankenstein_signal:
        power_chans.append(1 / (2 * len(channel) + 1) * np.sum(np.square(np.array(channel))))

    highest_channels = np.argpartition(power_chans, -4)[-4:]
    highest_data = np.array(frankenstein_signal)[highest_channels]

    return highest_data


def print_stats(fran_corr_1, fran_corr, fran_corr_2):
    print(str(time) + str(shapiro(fran_corr_1)) + str(shapiro(fran_corr)) + str(shapiro(fran_corr_2)))
    print(str(np.std(fran_corr_1)) + " " + str(np.std(fran_corr)) + " " + str(np.std(fran_corr_2)))
    print(str(np.mean(fran_corr_1)) + " " + str(np.mean(fran_corr)) + " " + str(np.mean(fran_corr_2)))
    print(str(np.median(fran_corr_1)) + " " + str(np.median(fran_corr)) + " " + str(np.median(fran_corr_2)))
    print(
        str(np.std(fran_corr_1) / np.mean(fran_corr_1)) + " " + str(np.std(fran_corr) / np.mean(fran_corr)) + " " + str(
            np.std(fran_corr_2) / np.mean(fran_corr_2)))


def find_hist(time_signal):
    time_signal = pd.DataFrame(time_signal).T.corr()
    time_signal.to_numpy()[np.diag_indices(23)] = 0
    time_signal = np.triu(time_signal, -1).flatten()[np.triu(time_signal, -1).flatten() != 0]
    return time_signal


def update(val):
    window = windowSlider.val
    time = timeSlider.val
    highest_data = max_channel_power(frankenstein_signal, int(time), int(window))
    fran_corr, fran_drop = calculate_window([frankenstein_signal], int(window), int(time))
    fran_corr_1, fran_drop_1 = calculate_window([frankenstein_signal], int(window), int(time - span_time))
    fran_corr_2, fran_drop_2 = calculate_window([frankenstein_signal], int(window), int(time + span_time))

    l_nor.set_data(A=fran_corr)
    l_2_nor.set_data(A=fran_corr_1)
    l_3_nor.set_data(A=fran_corr_2)

    o_data.set_segments([[[int(time), -1000], [int(time), 1000]], [[int(time+window), -1000], [int(time+window), 1000]]])
    ax_1.cla()
    ax_2.cla()
    ax_3.cla()

    fran_corr_1_ = fran_corr_1
    fran_corr_1_.to_numpy()[np.diag_indices(23)] = 0
    fran_corr_1_ = np.triu(fran_corr_1_, -1).flatten()[np.triu(fran_corr_1_, -1).flatten() != 0]
    ax_1.hist(fran_corr_1_, 20)

    fran_corr_ = fran_corr
    fran_corr_.to_numpy()[np.diag_indices(23)] = 0
    fran_corr_ = np.triu(fran_corr_, -1).flatten()[np.triu(fran_corr_, -1).flatten() != 0]
    ax_2.hist(fran_corr_, 20)

    fran_corr_2_ = fran_corr_2
    fran_corr_2_.to_numpy()[np.diag_indices(23)] = 0
    fran_corr_2_= np.triu(fran_corr_2_, -1).flatten()[np.triu(fran_corr_2_, -1).flatten() != 0]
    ax_3.hist(fran_corr_2_, 20)

    print_stats(fran_corr_1_, fran_corr_, fran_corr_2_)

window = 5 # Seconds
time = 15
patients = np.arange(2, 3)
span_time = 5

seizures_data, normal_data = load_correlations(patients)
frankenstein_signal = create_random_signal(seizures_data, normal_data, span_time)
highest_data = max_channel_power(frankenstein_signal, time, window)
fran_corr, fran_drop = calculate_window([frankenstein_signal], window, time)
fran_corr_1, fran_drop_1 = calculate_window([frankenstein_signal], window, time-span_time)
fran_corr_2, fran_drop_2 = calculate_window([frankenstein_signal], window, time+span_time)

x_axis = np.linspace(0, window*len(seizures_data)*2, int(len(frankenstein_signal[0])/(256*span_time)))
print(len(x_axis))
y_axis = np.resize([750, 0], len(x_axis))
x_axis_2 = np.linspace(0, len(seizures_data)*2*span_time, len(seizures_data)*2*span_time*256)

fig = plt.figure()
gs = gridspec.GridSpec(3, 3)
plt.subplots_adjust(bottom=0.25)

ax = fig.add_subplot(gs[0, 0])
ax.set_title("Correlation, -5s")
l_2_nor = ax.imshow(fran_corr_1, origin="lower", cmap="hot", interpolation="nearest")

ax = fig.add_subplot(gs[0, 1])
ax.set_title("Correlation")
l_nor = ax.imshow(fran_corr, origin="lower", cmap="hot", interpolation="nearest")

ax = fig.add_subplot(gs[0, 2])
ax.set_title("Correlation, +5s")
l_3_nor = ax.imshow(fran_corr_2, origin="lower", cmap="hot", interpolation="nearest")

ax = fig.add_subplot(gs[2, :])
ax.set_title("Signals")
print(len(x_axis_2))
s_data = ax.plot(x_axis_2, highest_data.T)
o_data = ax.vlines([time, time+window], -1000, 1000)
g_data = ax.plot(x_axis, y_axis, drawstyle="steps")

fran_corr_1_ = fran_corr_1
fran_corr_1_.to_numpy()[np.diag_indices(23)] = 0
fran_corr_1_ = np.triu(fran_corr_1_, -1).flatten()[np.triu(fran_corr_1_, -1).flatten() != 0]
ax_1 = fig.add_subplot(gs[1, 0])
ax_1.set_title("Histogram, -5s")
h_1 = ax_1.hist(fran_corr_1_, 200)
fit_1 = stats.norm.pdf(sorted(fran_corr_1_), np.mean(fran_corr_1_), np.std(fran_corr_1_))
ax_1.plot(sorted(fran_corr_1_), fit_1)
ax_1.text(0.8, 10, "Mean: " + str(np.mean(fran_corr_1_)))
ax_1.text(0.8, 5, "Std: " + str(np.std(fran_corr_1_)))

fran_corr_ = fran_corr
fran_corr_.to_numpy()[np.diag_indices(23)] = 0
fran_corr_ = np.triu(fran_corr_, -1).flatten()[np.triu(fran_corr_, -1).flatten() != 0]
ax_2 = fig.add_subplot(gs[1, 1])
ax_2.set_title("Histogram")
h_2 = ax_2.hist(fran_corr_, 20)
fit_2 = stats.norm.pdf(sorted(fran_corr_), np.mean(fran_corr_), np.std(fran_corr_))
ax_2.plot(sorted(fran_corr_), fit_2)
ax_2.text(0.8, 20, "Mean: " + str(np.mean(fran_corr_)))
ax_2.text(0.8, 15, "Std: " + str(np.std(fran_corr_)))


fran_corr_2_ = fran_corr_2
fran_corr_2_.to_numpy()[np.diag_indices(23)] = 0
fran_corr_2_ = np.triu(fran_corr_2_, -1).flatten()[np.triu(fran_corr_2_, -1).flatten() != 0]
ax_3 = fig.add_subplot(gs[1, 2])
ax_3.set_title("Histogram, +5s")
h_3 = ax_3.hist(fran_corr_2_, 20)
fit_3 = stats.norm.pdf(sorted(fran_corr_2_), np.mean(fran_corr_2_), np.std(fran_corr_2_))
ax_3.plot(sorted(fran_corr_2_), fit_3)
ax_3.text(0.8, 20, "Mean: " + str(np.mean(fran_corr_2_)))
ax_3.text(0.8, 15, "Std: " + str(np.std(fran_corr_2_)))
print_stats(fran_corr_1_, fran_corr_, fran_corr_2_)

axWindow = plt.axes([0.15, 0.1, 0.65, 0.03])
axTime = plt.axes([0.15, 0.15, 0.65, 0.03])

windowSlider = Slider(axWindow, 'Window', 1, 20, valinit=span_time, valstep=1)
timeSlider = Slider(axTime, 'Time', 0, 30, valinit=span_time, valstep=span_time)

windowSlider.on_changed(update)
timeSlider.on_changed(update)

# fig, (ax1, ax2, ax3) = plt.subplots(3)
#
# stats.probplot(fran_corr_1_, dist="norm", plot=ax1)
# stats.probplot(fran_corr_, dist="norm", plot=ax2)
# stats.probplot(fran_corr_2_, dist="norm", plot=ax3)

fig, (ax1, ax2) = plt.subplots(2)

all_selected_seiz = [(find_hist(frankenstein_signal[:, span_time*i*256:(i*span_time+span_time)*256]))for i in range(0, 2*len(seizures_data), 2)]
all_selected_nor = [(find_hist(frankenstein_signal[:, span_time*i*256:((span_time * i)+span_time)*256]))for i in range(1, 2*len(seizures_data), 2)]

all_selected_seiz = np.array(all_selected_seiz).flatten()
all_selected_nor = np.array(all_selected_nor).flatten()

nor_fit = stats.norm.pdf(np.sort(all_selected_nor), np.mean(all_selected_nor), np.std(all_selected_nor))
seiz_fit = stats.norm.pdf(np.sort(all_selected_seiz), np.mean(all_selected_seiz), np.std(all_selected_seiz))

ax1.plot(np.sort(all_selected_seiz), seiz_fit)
ax1.hist(all_selected_seiz, 200)
print("Mean: " + str(np.mean(all_selected_seiz)) + " Std: " + str(np.std(all_selected_seiz)))

ax2.plot(np.sort(all_selected_nor), nor_fit)
ax2.hist(all_selected_nor, 200)
print("Mean: " + str(np.mean(all_selected_nor)) + " Std: " + str(np.std(all_selected_nor)))

plt.show()
