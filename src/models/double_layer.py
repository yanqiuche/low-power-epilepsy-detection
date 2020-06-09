import torch.nn as nn
import torch.nn.functional as F
import torch

# class DoubleLayer(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.linears = nn.ModuleList([nn.Linear(input_size, 1, bias=True) for i in range(23)])
#         self.linear_2 = nn.Linear(23, 1, bias=True)
#         # self.relus = nn.ModuleList([nn.ReLU() for i in range(23)])
#         # self.relu_1 = nn.ReLU()
#         # self.sig = nn.Sigmoid()
#
#     def forward(self, x):
#         bs = x.size(0)
#         out = torch.zeros([23, bs, 1])
#         for i, l in enumerate(self.linears):
#             out[i] = l(x[:, i, :])
#         out = torch.transpose(out[:, :, 0], 0, 1)
#         out = self.linear_2(out.double())
#
#         return out


class DoubleLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(input_size, 1, bias=True) for i in range(23)])
        self.linear_2 = nn.Linear(23, 1, bias=True)

        # self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.ModuleList([nn.BatchNorm1d(1024) for i in range(23)])
        self.batchnorm2 = nn.BatchNorm1d(23)
        # self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        bs = x.size(0)
        out = torch.zeros([23, bs, 1])
        for i, l in enumerate(self.linears):
            out[i] = l(self.batchnorm1[i](x[:, i, :]))
        out = torch.transpose(out[:, :, 0], 0, 1)
        out = self.batchnorm2(out.double())
        out = self.dropout(out)
        out = self.linear_2(out)

        return out
