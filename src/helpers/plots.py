import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from src.models.helpers import load_correlations, calculate_window


def update(val):
    window = windowSlider.val
    time = timeSlider.val

    nor_corr, nor_drop = calculate_window(seizures_data, int(window), int(time))
    seiz_corr, seiz_drop = calculate_window(normal_data, int(window), int(time))
    print("Total size seizures: " + str(seiz_corr.to_numpy().sum()) + "Total size normal: " + str(
        nor_corr.to_numpy().sum()))

    if nor_drop or seiz_drop:
        print("Dropped seizures: " + str(nor_drop) + "Dropped seizures: " + str(seiz_drop))

    l_seiz.set_data(A=seiz_corr)
    l_nor.set_data(A=nor_corr)


window = 1 # Seconds
time = 1
patients = np.arange(1, 10)
# patients = [10, 11]

seizures_data, normal_data = load_correlations(patients)
nor_corr, nor_drop = calculate_window(seizures_data, window, time)
seiz_corr, seiz_drop = calculate_window(normal_data, window, time)
print("Total size seizures: " + str(seiz_corr.to_numpy().sum()) + "Total size normal: " + str(nor_corr.to_numpy().sum()))

fig, axes = plt.subplots(nrows=1, ncols=2)
plt.subplots_adjust(bottom=0.25)
axes[0].set_title("Seizure correlation")
l_nor = axes[0].imshow(nor_corr, origin="lower", cmap="hot", interpolation="nearest")

axes[1].set_title("Normal correlation")
l_seiz = axes[1].imshow(seiz_corr, origin="lower", cmap="hot", interpolation="nearest")

axWindow = plt.axes([0.15, 0.1, 0.65, 0.03])
axTime = plt.axes([0.15, 0.15, 0.65, 0.03])

windowSlider = Slider(axWindow, 'Window', 1, 20, valinit=5, valstep=1)
timeSlider = Slider(axTime, 'Time', 0, 30, valinit=0, valstep=1)

windowSlider.on_changed(update)
timeSlider.on_changed(update)
plt.show()





