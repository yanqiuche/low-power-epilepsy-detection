# convolutional network model we will train to detect patterns in readings. For more information see my tutorial here.
# Found at; https://github.com/SamLynnEvans/EEG-grasp-and-lift
# Training binary classification; https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89

import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.models.train import train
from src.models.load_data import load_data
from src.models.cnn_model import twod_convmodel

window_size = 1024
sample_spacing = 256
channels = 23
bs = 200
lr = 0.05 # 0.0005  # 67% lr = 0.00075 # 64%
epochs = 50
train_ratio = 0.8
type = 3
cnn = 1

if cnn:
    overlap = 0.5
    seconds = 30

eeg_processed_folder = "/home/jmsvanrijn/Documents/Afstuderen/Code/low-power-epilepsy-detection/data/processed/"

writer = SummaryWriter()
train_loader, valid_loader = load_data(eeg_processed_folder, window_size, bs, type, train_ratio)

# Possibly need to normalize first z = (x-u)/s
# epilepsy_model = DoubleLayer(window_size, window_size, channels).double()
# epilepsy_model = convmodel(window_size, 1).double()
epilepsy_model = twod_convmodel().double()
optimizer = optim.Adam(epilepsy_model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()
train(epilepsy_model, train_loader, valid_loader, epochs, criterion, optimizer, writer)

writer.close()
