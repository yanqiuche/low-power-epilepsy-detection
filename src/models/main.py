# convolutional network model we will train to detect patterns in readings. For more information see my tutorial here.
# Found at; https://github.com/SamLynnEvans/EEG-grasp-and-lift

import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from epilepsy_data_loader import EpilepsyData
from single_layer import SingleLayerNN
from double_layer import DoubleLayer
from cnn_model import convmodel
from learn_rate import find_lr
from train import train
from visualize import visualize_train_results
import random

window_size = 1024
sample_spacing = 256
bs = 400
lr = 0.0005  #0.0005  # 67% lr = 0.00075 # 64%
epochs = 10
train_ratio = 0.8
save = False
EEG_DATA = "/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/"

# Create a array seizures, 1= seizure, 0= normal
seizure_files_2 = [(str(seizure), str(1)) for seizure in sorted(Path(EEG_DATA).glob('seizures/*.edf'))]
normal_files_2 = [(str(normal), str(0)) for normal in sorted(Path(EEG_DATA).glob('normal/*.edf'))]
all_files = seizure_files_2 + normal_files_2
all_files = random.sample(all_files, len(all_files))
train_data = EpilepsyData(all_files[:round(train_ratio*len(all_files))], window_size)
valid_data = EpilepsyData(all_files[round(train_ratio*len(all_files)):], window_size)
train_loader = DataLoader(train_data, batch_size=bs, shuffle=True)
valid_data = DataLoader(valid_data, batch_size=bs, shuffle=True)

epilepsy_model_1 = DoubleLayer(window_size, 1).double()
optimizer_1 = optim.SGD(epilepsy_model_1.parameters(), lr=lr, momentum=0.9)

# epilepsy_model_2 = SingleLayerNN(23*window_size, 1).double()
# optimizer_2 = optim.SGD(epilepsy_model_2.parameters(), lr=lr, momentum=0.9)
criterion = nn.BCELoss()
# results = train(train_loader, valid_data, epochs, optimizer_2, epilepsy_model_2)
valid, results = train(epilepsy_model_1, train_loader, valid_data, epochs, criterion, optimizer_1)

visualize_train_results(valid, results)

module = epilepsy_model_1
print(list(module.named_parameters()))

# torch.save(epilepsy_model_1.state_dict(), "./models/model_2.pth")
