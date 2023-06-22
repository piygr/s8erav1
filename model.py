import torch.nn as nn
import torch.nn.functional as F


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
