# CNNLSTM
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

        self.conv1 = nn.Conv2d(
            3,
            5,
            kernel_size=11,
            stride=(1, 1),
            padding=(1, 1),
            bias=False)
        self.bn1 = nn.BatchNorm2d(5)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1,1))

        self.conv2 = nn.Conv2d(
            5,
            10,
            kernel_size=6,
            stride=(1, 1),
            padding=(1, 1),
            bias=False)
        self.bn2 = nn.BatchNorm2d(10)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(1,1))

        self.avgpool = nn.AvgPool2d(
            (92, 125), stride=1)

        self.lstm = nn.LSTMCell(500, 500)
        self.fc1 = nn.Linear(500, 9)
        self.soft_max = nn.Softmax()

        self.hx = torch.randn(1,500)
        self.cx = torch.randn(1, 500)
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
        x = torch.tanh(x)
        x = self.maxpool1(x)

        print("size before conv2{}".format(x.size()))

        x = torch.squeeze(x)
        print(x.size())
        x = self.conv2(x)
        #x = self.bn2(x)
        #x = self.relu2(x)
        x = self.maxpool2(x)
        x = torch.tanh(x)

        print("size before avgpool{}".format(x.size()))

        #x = self.avgpool(x)
        #print(x.size())

        x = x.view(-1, 50)
        print(x.size())
        self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        print(self.hx.size())
        x = self.fc1(self.hx)

        print(x.size())

        x = self.soft_max(x)
        print(x.size())
        return x


""" for MCCNN
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
        self.bn1 = nn.BatchNorm3d(20)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 5, 5), stride=(1,1,1))

        self.conv2 = nn.Conv2d(
            20,
            50,
            kernel_size=3,
            stride=(1, 1),
            padding=(1, 1),
            bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(5, 5), stride=(1,1))

        self.avgpool = nn.AvgPool2d(
            (92, 125), stride=1)

        self.fc1 = nn.Linear(50, 500)
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
        x = self.bn1(x)
        #x = self.relu1(x)
        #x = torch.tanh(x)
        x = self.maxpool1(x)

        print("size before conv2{}".format(x.size()))

        x = torch.squeeze(x)
        print(x.size())
        x = self.conv2(x)
        x = self.bn2(x)
        #x = self.relu2(x)
        x = self.maxpool2(x)
        print(x.size())

        # sum over temporal dimension
        #x = x.sum(2)

        print("size before avgpool{}".format(x.size()))

        x = self.avgpool(x)
        print(x.size())

        x = x.view(-1, 50)
        print(x.size())
        x = self.fc1(x)
        x = self.fc2(x)
        print(x.size())

        x = self.soft_max(x)
        print(x.size())
        return x """""


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class myModel(nn.Module):

    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=9):

        super(myModel, self).__init__()

        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1,1,1))

        self.conv2 = nn.Conv3d(
            64,
            128,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False)
        self.bn2 = nn.BatchNorm3d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1,1,1))

        self.conv3a = nn.Conv3d(
            128,
            256,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False)
        self.conv3b = nn.Conv3d(
            256,
            256,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False)
        self.bn3 = nn.BatchNorm3d(256)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 1, 1))

        self.conv4a = nn.Conv3d(
            256,
            512,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False)
        self.conv4b = nn.Conv3d(
            512,
            512,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False)

        self.bn4 = nn.BatchNorm3d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 1, 1))

        self.conv5 = nn.Conv3d(
            512,
            512,
            kernel_size=3,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=False)
        self.bn5 = nn.BatchNorm3d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 1, 1))

        '''self.conv3 = nn.Conv2d(
            128,
            64,
            kernel_size=3,
            stride=(2, 2),
            padding=(2, 2),
            bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2,2), padding=(1,1))

        self.conv4 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            stride=(2, 2),
            padding=(3, 3),
            bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2,2), padding=(1,1))'''

        last_duration = int(math.ceil(sample_duration / 1.25))
        last_size = int(math.ceil(sample_size / 86))
        self.avgpool = nn.AvgPool2d(
            (95, 128), stride=1)

        self.fc1 = nn.Linear(512, num_classes)
        self.fc2 = nn.Linear(4096, num_classes)
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

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.bn3(x)
        x = self.maxpool3(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        x = self.bn4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        #x = self.conv5(x)
        x = self.maxpool5(x)
        print(x.size())
        # sum over channel dimension
        x = x.sum(2)

        '''print('size after sum {}'.format(x.size()))
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        print(x.size())

        print(x.size())
        x = self.conv4(x)
        #x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)'''

        print("size before avgpool{}".format(x.size()))

        x = self.avgpool(x)
        print(x.size())

        x = x.view(-1, 512)
        print(x.size())
        x = self.fc1(x)
        #x = self.fc2(x)
        print(x.size())

        x = self.soft_max(x)

        return x"""

