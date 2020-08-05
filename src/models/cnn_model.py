import torch.nn as nn


class convmodel(nn.Module):
    def __init__(self, window_size, out_classes, drop=0.5, d_linear=124):
        super().__init__()

        self.conv2 = nn.Conv1d(23, 46, kernel_size=3, padding=0, stride=1)
        self.bn = nn.BatchNorm1d(46)
        self.pool = nn.MaxPool1d(2, stride=2)
        # self.linear1 = nn.Linear(5842, d_linear)

        self.linear1 = nn.Linear((round(window_size/2)-1)*46, d_linear)

        self.linear3 = nn.Linear(d_linear, out_classes)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)
        self.dropout3 = nn.Dropout(drop)

        self.conv = nn.Sequential(self.conv2, nn.ReLU(inplace=True), self.bn, self.pool, self.dropout1)
        self.dense = nn.Sequential(self.linear1, nn.ReLU(inplace=True), self.dropout2, self.dropout3, self.linear3)

    def forward(self, x):
        bs = x.size(0)
        x = self.conv(x)
        x = x.view(bs, -1)
        output = self.dense(x)

        return output


class twod_convmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(23*23, 10*10, bias=True)
        self.linear2 = nn.Linear(10*10, 1, bias=True)
        self.linear3 = nn.Linear(5*5, 1, bias=True)
        self.batch1 = nn.BatchNorm1d(23*23)
        self.batch2 = nn.BatchNorm1d(10*10)
        self.batch3 = nn.BatchNorm1d(5*5)

    def forward(self, x):
        bs = x.size(0)
        out = x.view(bs, -1)
        out = self.batch1(out)
        out = self.linear1(out)
        out = self.batch2(out)
        out = self.linear2(out)
        # out = self.batch3(out)
        # out = self.linear3(out)
        return out
