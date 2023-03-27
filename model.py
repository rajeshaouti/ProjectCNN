import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # INPUT BLOCK
        self.conv1 = nn.Sequential(
             nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = (3,3), padding= 0), # 28>26 | 3
             nn.ReLU()
        )

        # CONVOLUTION BLOCK 1
        self.conv2 = nn.Sequential(
             nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding= 0), # 26>24 | 5
             nn.ReLU()
        )
        self.conv3 = nn.Sequential(
             nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding= 0), # 24>22 | 7
             nn.ReLU()
        )

        # TRANSITION BLOCK
        self.pool1 = nn.MaxPool2d(2,2)                                                        # 22>11 | 14
        self.conv4 = nn.Sequential(
             nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = (1,1), padding= 0), # 11>11 | 14
             nn.ReLU()
        )

        # CONVOLUTION BLOCK 2
        self.conv5 = nn.Sequential(
             nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3,3), padding= 0), # 11>9 | 16
             nn.ReLU()
        )
        self.conv6 = nn.Sequential(
             nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3,3), padding= 0), # 9>7 | 18
             nn.ReLU()
        )

        # OUTPUT BLOCK
        self.conv7 = nn.Sequential(
              nn.Conv2d(in_channels = 128, out_channels = 10, kernel_size = (1,1), padding= 0), # 7>7 | 18
        )
        self.conv8 = nn.Sequential(
              nn.Conv2d(in_channels = 10, out_channels = 10, kernel_size = (7,7), padding= 0), # 7>1 | 24
        )

    def forward(self, x):

      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      x = self.pool1(x)
      x = self.conv4(x)
      x = self.conv5(x)
      x = self.conv6(x)
      x = self.conv7(x)
      x = self.conv8(x)
      
      x = x.view(-1,10)
      return x
      # return F.log_softmax(x, dim=-1)
