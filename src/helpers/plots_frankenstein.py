import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from src.models.helpers import load_correlations, calculate_window
import random


def create_random_signal(seizures_data, normal_data):
    span_time = 5  # in seconds

    s_time = random.randint(0, int(((len(seizures_data[0][0][0]) / 256) - span_time)))
    seizures = np.array(seizures_data[0][0][:, s_time * 256:(s_time + span_time) * 256])
    normals = np.array(normal_data[0][0][:,  s_time * 256:(s_time + span_time) * 256])

    frankenstein_signal = np.hstack((seizures, normals))

    for i in range(1, len(seizures_data)):
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

    highest_channels = np.argpartition(power_chans, -10)[-10:]
    highest_data = np.array(frankenstein_signal)[highest_channels]

    return highest_data


def update(val):
    window = windowSlider.val
    time = timeSlider.val
    highest_data = max_channel_power(frankenstein_signal, int(time), int(window))
    fran_corr, fran_drop = calculate_window([frankenstein_signal], int(window), int(time))

    # o_data.remove()
    # oo_data.remove()

    l_nor.set_data(A=fran_corr)
    # for i in range(len(highest_data)):
    #     s_data[i].set_ydata(highest_data[i].T)

    o_data.set_segments([[[int(time), -1500], [int(time), 1500]], [[int(time+window), -1500], [int(time+window), 1500]]])

    # axes[1].vlines(time, -1500, 1500)
    # axes[1].vlines(time + window, -1500, 1500)

window = 5 # Seconds
time = 5
patients = np.arange(2, 3)
# patients = [10, 11]

seizures_data, normal_data = load_correlations(patients)
frankenstein_signal = create_random_signal(seizures_data, normal_data)
highest_data = max_channel_power(frankenstein_signal, time, window)
fran_corr, fran_drop = calculate_window([frankenstein_signal], window, time)

x_axis = np.linspace(0, 5*window, int(len(frankenstein_signal[0])/1280))
y_axis = np.resize([750, 0], len(x_axis))
x_axis_2 = np.linspace(0, 6*window, 6*window*256)

fig, axes = plt.subplots(nrows=3)
plt.subplots_adjust(bottom=0.25)
axes[0].set_title("Seizure correlation")
l_nor = axes[0].imshow(fran_corr, origin="lower", cmap="hot", interpolation="nearest")
fig.colorbar(l_nor)

axes[1].set_title("Signals")
# s_data = axes[1].plot(x_axis_2, highest_data.T)
s_data = axes[1].plot(x_axis_2, highest_data.T)
g_data = axes[1].plot(x_axis, y_axis, drawstyle="steps")
o_data = axes[1].vlines([time, time+window], -1500, 1500)
# oo_data = axes[1].vlines(time+window, -1500, 1500)

axWindow = plt.axes([0.15, 0.1, 0.65, 0.03])
axTime = plt.axes([0.15, 0.15, 0.65, 0.03])

windowSlider = Slider(axWindow, 'Window', 1, 20, valinit=5, valstep=1)
timeSlider = Slider(axTime, 'Time', 0, 30, valinit=5, valstep=1)

windowSlider.on_changed(update)
timeSlider.on_changed(update)

# print(np.concatenate(frankenstein_signal).ravel())
# axes[2] = plt.hist(np.concatenate(frankenstein_signal).ravel(), 20)

plt.show()





