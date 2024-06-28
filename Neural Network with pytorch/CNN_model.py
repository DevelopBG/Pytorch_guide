
from torch import nn
import torch.nn.functional as F
import torch






##Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.batchnrm1= nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.batchnrm2= nn.BatchNorm2d(16)
        # self.fc1 = nn.Linear(16 * 8* 8,1024)
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)
        # self.fc2= nn.Linear(1024,num_classes)

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.conv1(x))
        x= self.batchnrm1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x= self.batchnrm2(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc1(x)
        return x