from matplotlib import pyplot as plt
from src.models.learn_rate import find_lr
from pathlib import Path
from pyedflib import highlevel
import numpy as np

def visualize_train_results(valid, results):
    epochs = int(results.iloc[-1]["epoch"])
    iterations = int(results.iloc[-1]["iteration"])
    tr, tco, tse, tsp = [], [], [], []

    print(results.to_string)

    for epoch in range(epochs):
        fn = results.iloc[epoch * iterations:(epoch + 1) * iterations]["fn"].sum()
        fp = results.iloc[epoch * iterations:(epoch + 1) * iterations]["fp"].sum()
        tn = results.iloc[epoch * iterations:(epoch + 1) * iterations]["tn"].sum()
        tp = results.iloc[epoch * iterations:(epoch + 1) * iterations]["tp"].sum()
        tr.append(results.iloc[((epoch + 1) * iterations)-1]["loss"])
        tco.append((tn + tp) / (tp + tn + fp + fn))
        tse.append(tp / (tp + fn))
        tsp.append(tn / (tn + fp))

    plt.plot(valid, label="Validator")
    plt.plot(tco, label="Correctness")
    plt.plot(tse, label="Sensitivity")
    plt.plot(tsp, label="Specificity")
    plt.legend()
    plt.show()

    plt.plot(tr, label="Loss")
    plt.legend()
    plt.show()


def data_length(folder):
    seizure_files = [(str(seizure)) for seizure in sorted(Path(folder).glob('seizures/*.edf'))]
    normal_files = [(str(normal)) for normal in sorted(Path(folder).glob('normal/*.edf'))]
    print("Number of files with seizures: " + str(len(seizure_files)))
    print("Number of files without siezures: " + str(len(normal_files)))

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


def find_learning_rate():
    log, losses = find_lr(model, F.binary_cross_entropy, optimizer, train_loader, init_value=1e-8, final_value=10e-4, device="cpu")
    total_correctness, total_sensitivity, total_specificity, total_loss = train(train_loader, 1, optimizer_1, epilepsy_model_1)


def profile_per_file():
    pass


# data_length("/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/")