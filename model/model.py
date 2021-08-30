import torch.nn as nn


class ResBlock_original(nn.Module):
    '''
        x --> CONV --> BN --> ReLU --> CONV --> BN -(y)->[add]--> ReLU --> z
          |                                            /
          |--------------------x----------------------/            (if x.shape == y.shape)
          |                                          /
          |----- x --> [1x1]CONV --> x_update ------/              (if x.shape != y.shape)
    '''

    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock_original, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.conv_extend_dimension = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)

        y = nn.functional.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if (x.shape == y.shape):
            z = x
        else:
            z = self.conv_extend_dimension(x)

        z = z + y
        z = nn.functional.relu(z)
        return z


class ResBlock_iden(nn.Module):
    '''
        x --> CONV --> BN --> ReLU --> CONV --> BN -(y)->[add]--> ReLU --> z
          |                                            /
          |--------------------x----------------------/            (if x.shape == y.shape)
          |                                          /
          |----- x --> [1x1]CONV --> x_update ------/              (if x.shape != y.shape)
    '''

    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock_iden, self).__init__()

        self.bn0 = nn.BatchNorm2d(num_features=in_channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.conv_extend_dimension = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=2)

    def forward(self, x):
        y = self.bn0(x)
        y = nn.functional.relu(y)

        y = self.conv1(y)
        y = self.bn1(y)

        y = nn.functional.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)

        if (x.shape == y.shape):
            z = x
        else:
            z = self.conv_extend_dimension(x)

        z = z + y
        z = nn.functional.relu(z)
        return z


class PlainBlock(nn.Module):
    '''
        x --> CONV --> BN --> ReLU --> CONV --> BN --> ReLU --> z
    '''

    def __init__(self, in_channels, out_channels, stride):
        super(PlainBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = nn.functional.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = nn.functional.relu(y)
        return y

class BaseNet(nn.Module):

    def __init__(self, Block, n):
        super(BaseNet, self).__init__()

        self.n = n
        self.Block = Block
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(num_features=16)

        self.conv_hidden = self._make_layers(n)

        self.avgPool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = nn.functional.relu(x)

        x = self.conv_hidden(x)

        x = self.avgPool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layers(self, n):
        layers = []
        in_channels = 16
        out_channels = 16
        stride = 1
        for i in range(3):
            for j in range(self.n):
                if (i > 0 and j == 0):
                    in_channels = out_channels
                    out_channels *= 2
                    stride = 2

                layers += [self.Block(in_channels, out_channels, stride)]

                stride = 1
                in_channels = out_channels
        return nn.Sequential(*layers)


def ResNet_original(n):
    return BaseNet(ResBlock_original, n)


def ResNet_identity(n):
    return BaseNet(ResBlock_iden, n)


def PlainNet(n):
    return BaseNet(PlainBlock, n)
