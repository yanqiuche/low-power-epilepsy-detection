# convolutional network model we will train to detect patterns in readings. For more information see my tutorial here.
# Found at; https://github.com/SamLynnEvans/EEG-grasp-and-lift
# Training binary classification; https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89

import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.models.double_layer import DoubleLayer
from src.models.train import train
from src.models.load_data import load_data

window_size = 1024
sample_spacing = 256
bs = 200
lr = 0.001  #0.0005  # 67% lr = 0.00075 # 64%
epochs = 20
train_ratio = 0.8
type = 0
eeg_processed_folder = "/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/"

writer = SummaryWriter()
train_loader, valid_loader = load_data(eeg_processed_folder, window_size, bs, type, train_ratio)

# Possibly need to noramilze first z = (x-u)/s
epilepsy_model_1 = DoubleLayer(window_size, 1).double()
optimizer_1 = optim.Adam(epilepsy_model_1.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()
train(epilepsy_model_1, train_loader, valid_loader, epochs, criterion, optimizer_1, writer)

writer.close()
