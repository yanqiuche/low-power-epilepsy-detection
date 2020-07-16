import torch.nn as nn

class ltsmModel(nn.Module):
    def __init__(self, input_size, windows_size_channels):
        super.__init__()
        self.lstm = nn.LSTM()

    def forward(self, x):