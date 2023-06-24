import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


NO_GROUPS = 4

class Block(nn.Module):
    def __init__(self, input_channel, output_channel, padding=1, norm='bn', drop=0.01):

        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=padding)

        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(output_channel)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(NO_GROUPS, output_channel)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, output_channel)

        self.drop1 = nn.Dropout2d(drop)

        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=padding)

        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(output_channel)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(NO_GROUPS, output_channel)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, output_channel)

        self.drop2 = nn.Dropout2d(drop)

        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=padding)

        if norm == 'bn':
            self.n3 = nn.BatchNorm2d(output_channel)
        elif norm == 'gn':
            self.n3 = nn.GroupNorm(NO_GROUPS, output_channel)
        elif norm == 'ln':
            self.n3 = nn.GroupNorm(1, output_channel)

        self.drop3 = nn.Dropout2d(drop)


    def __call__(self, x, layers=2, last=False):
        x = self.conv1(x)
        x = self.n1(x)
        x = F.relu(x)

        x1 = torch.Tensor(x)
        x = self.drop1(x)


        if layers >= 2:
            x = self.conv2(x)
            x = self.n2(x)
            x = F.relu(x)
            x = self.drop2(x)

        if layers == 3:
            x = x1 + self.conv3(x)
            x = self.n3(x)
            x = F.relu(x)
            x = self.drop3(x)

        return x


class TransitionBlock(nn.Module):
    def __init__(self, output_channel, pooling=True):
        super(TransitionBlock, self).__init__()

        self.tconv = nn.Conv2d(output_channel, int(output_channel/2), 1)

        self.pooling = pooling

        self.pool = nn.MaxPool2d(2, 2)

    def __call__(self, x):
        x = self.tconv(x)

        if self.pooling:
            x = self.pool(x)

        return x


class S8Model(nn.Module):
    def __init__(self, base_channels, norm, drop=0.01):
        super(S8Model, self).__init__()

        self.block1 = Block(3, base_channels, padding=1, norm=norm, drop=drop)
        self.tblock1 = TransitionBlock(base_channels, pooling=True)

        self.block2 = Block(int(base_channels/2), base_channels*2, padding=1, norm=norm, drop=drop)
        self.tblock2 = TransitionBlock(base_channels*2, pooling=True)

        self.block3 = Block(base_channels, base_channels*2, padding=1, norm=norm, drop=drop)
        self.gap = nn.AvgPool2d(8)

        self.linear = nn.Conv2d(base_channels*2, 10, 1)

    def forward(self, x):
        x = self.block1(x, layers=2)
        x = self.tblock1(x)
        #print(x.size())

        x = self.block2(x, layers=3)
        x = self.tblock2(x)
        #print(x.size())

        x = self.block3(x, layers=3)
        #print(x.size())

        x = self.gap(x)
        x = self.linear(x)

        x = x.view(x.size(0), 10)

        return F.log_softmax(x, dim=1)




'''
Assignment S7
'''
class Model1(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Model1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),    # rf 3  -> 26x26
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 16, 3),  # rf 5   -> 24x24
            nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(2, 2) #rf 6 -> 12x12

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3),    #rf 10 -> 10x10
            nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(2, 2) #rf 12 -> 5x5

        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3),       #rf 20 -> 3x3
            nn.ReLU(),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3),  # rf 28 -> 1x1
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 10, 1)
        )


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.pool2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), 10)

        return F.log_softmax(x, dim=1)


    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, verbose=True)


class Model2(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Model2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),    # rf 3  -> 28x28
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.Dropout2d(0.05)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),   #rf 5   -> 28x28
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.05)
        )

        self.pool1 = nn.MaxPool2d(2, 2) #rf 6 -> 12x12

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 12, 3),    #rf 10 -> 12x12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(0.05)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 12, 3),  # rf 14 -> 10x10
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(0.05),
        )

        self.pool2 = nn.MaxPool2d(2, 2) #rf 16 -> 5x5

        self.conv5 = nn.Sequential(
            nn.Conv2d(12, 16, 3),       #rf 24 -> 3x3
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, 3),      #rf 28 -> 1x1
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05)
        )

        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 10, 1)
        )


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), 10)

        return F.log_softmax(x, dim=1)



    def get_scheduler(self, optimizer):
        return optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, verbose=True)


class Model3(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Model3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),  # rf 3  -> 28x28
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.05)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),  # rf 5   -> 28x28
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.05)
        )

        self.pool1 = nn.MaxPool2d(2, 2)  # rf 6 -> 14x14

        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 12, 3),  # rf 10 -> 12x12
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout2d(0.05)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(12, 16, 3),  # rf 14 -> 10x10
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05),
        )

        self.pool2 = nn.MaxPool2d(2, 2)  # rf 16 -> 5x5

        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 20, 3),  # rf 24 -> 3x3
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.Dropout2d(0.05)
        )

        '''
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, 3),  # rf 28 -> 1x1
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.05)
        )


        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 10, 1)            # 10x1x1
        )'''

        self.gap = nn.Sequential(
            nn.AvgPool2d(3)  # 24 + (3-1)*4
        )

        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        # x = self.conv6(x)
        # x = self.conv7(x)
        x = self.gap(x)
        # print(x.size())
        x = x.view(x.size(0), 20)
        # print(x.size())
        x = self.fc(x)

        return F.log_softmax(x, dim=1)


    def get_scheduler(self, optimizer):
        # return optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1, verbose=True)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, threshold=0.001, threshold_mode='abs',
                                                    eps=0.001, verbose=True)


'''
Assignment S6 - Part 2
'''
class OptimizedNet(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(OptimizedNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), #=> 8x28x28
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.02),
            nn.Conv2d(8, 8, 3, padding=1), #=> 8x28x28
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout2d(0.02),
            nn.MaxPool2d(2, 2)             #=> 8x14x14
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), # => 16x14x14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.02),
            nn.Conv2d(16, 16, 3, padding=1),    # => 16x14x14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.02)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3),    # => 32x12x12
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.02),
            nn.MaxPool2d(2, 2),      # => 32x6x6
            nn.Conv2d(32, 32, 3),    # => 32x4x4
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.02)
        )
        self.conv_linear = nn.Conv2d(32, 10, 1) # => 10x4x4


    def forward(self, x):

        #print(x.size())         # => torch.Size([<2>, 1, 28, 28])
        x = self.conv1(x)

        #print(x.size())         # => torch.Size([<2>, 8, 14, 14])
        x = self.conv2(x)

        #print(x.size())         # => torch.Size([<2>, 16, 14, 14])
        x = self.conv3(x)

        #print(x.size())         # => torch.Size([<2>, 32, 4, 4])
        x = self.conv_linear(x)

        #print(x.size())         # => torch.Size([<2>, 10, 4, 4])
        x = F.avg_pool2d(x, 4)          # => 10x1x1

        #print(x.size())         # => torch.Size([<2>, 10, 1, 1])
        x = x.view(x.size(0), 10)

        #print(x.size())  # => torch.Size([<2>, 10])

        return F.log_softmax(x, dim=1)


'''
Assignment 5
'''
class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
