import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


class myModel(nn.Module):

    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=9):

        super(myModel, self).__init__()

        self.conv1 = nn.Conv3d(
            3,
            20,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1,1,1))

        self.conv2 = nn.Conv2d(
            20,
            50,
            kernel_size=3,
            stride=(1, 1),
            padding=(1, 1),
            bias=False)
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1,1))

        self.avgpool = nn.AvgPool2d(
            (5, 5), stride=1)

        self.fc1 = nn.Linear(4096, 500)
        self.fc2 = nn.Linear(500, 9)
        self.soft_max = nn.Softmax()

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')

    def forward(self, x):
        print(x.size())
        x = self.conv1(x)
        #x = self.bn1(x)
        #x = self.relu1(x)
        x = self.maxpool1(x)
        print(x.size())

        #print(x.size())
        x = self.conv2(x)
        #x = self.bn2(x)
        #x = self.relu2(x)
        x = self.maxpool2(x)
        print(x.size())

        # sum over temporal dimension
        x = x.sum(2)

        print("size before avgpool{}".format(x.size()))

        x = self.avgpool(x)
        print(x.size())

        x = x.view(-1, 500)
        print(x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        print(x.size())

        x = self.soft_max(x)

        return x