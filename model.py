import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 3, padding=0, bias=False)       # 28 > 26 | 3
        self.batch_norm1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 12, 3, padding=0, bias=False)      # 26 > 24 | 5
        self.batch_norm2 = nn.BatchNorm2d(12)
        self.conv3 = nn.Conv2d(12, 12, 3, padding=0, bias=False)      # 24 > 22 | 7
        self.batch_norm3 = nn.BatchNorm2d(12)
        self.conv4 = nn.Conv2d(12, 12, 3, padding=0, bias=False)      # 22 > 20 | 9
        self.batch_norm4 = nn.BatchNorm2d(12)

        self.pool1 = nn.MaxPool2d(2, 2)                               # 20 > 10 | 10

        self.conv5 = nn.Conv2d(12, 16, 3, padding=0, bias=False)      # 10 > 8 | 14
        self.batch_norm5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 16, 3, padding=0, bias=False)      #  8 > 6 | 18
        self.batch_norm6 = nn.BatchNorm2d(16)

        self.conv7 = nn.Conv2d(16, 10, 6, padding=0, bias=False)     # 6 > 1 | 28

    def forward(self, x):
        x = self.batch_norm1(F.relu(self.conv1(x)))
        x = self.batch_norm2(F.relu(self.conv2(x)))
        x = self.batch_norm3(F.relu(self.conv3(x)))
        x = self.batch_norm4(self.conv4(x))

        x = self.pool1(x)

        x = self.batch_norm5(F.relu(self.conv5(x)))
        x = self.batch_norm6(F.relu(self.conv6(x)))
        x = self.conv7(x)

        x = x.view(-1, 10)                                             # 1x1x10 > 10
        return F.log_softmax(x, dim=-1)