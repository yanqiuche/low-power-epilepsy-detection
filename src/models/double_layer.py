import torch.nn as nn
import torch

class DoubleLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, 1) for i in range(23)])
        self.linear_2 = nn.Linear(23, 1)
        # self.relus = nn.ModuleList([nn.ReLU() for i in range(23)])
        self.relu_1 = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        bs = x.size(0)
        out = torch.zeros([23, bs, 1])
        for i, l in enumerate(self.linears):
            out[i] = l(x[:, i, :])
        out = torch.transpose(out[:, :, 0], 0, 1)
        out = self.linear_2(out.double())

        return self.sig(out)
