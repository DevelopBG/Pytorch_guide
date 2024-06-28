import torch
import sys
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import (DataLoader)
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

in_channels = 3
num_classes = 10
learning_rate =  1e-3
batch_size = 1024
num_epoch = 5

''' trained model is suitable if we use same dataset for testing, but dataset is diferent
thn we need to apply some modifications. As we can see here output has 1000 classes but for cifar10 has 10.
So, we need to modify it'''
model = torchvision.models.vgg16(pretrained= True)
# print(model)
# sys.exit()

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return x

model1 = torchvision.models.vgg16(pretrained=False)
'''for network modification'''
# for param in model.parameters():
#     param.requires_grad = False
model1.avgpool = Identity() # changing avg pooling
model1.classifier = nn.Sequential(nn.Linear(512,10))
#
# print(model1)
model1.to(device)






train_dataset = datasets.CIFAR10(root = 'datasets/', train = True, transform= transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle = True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
for epoch in range(10):
    for i,j in tqdm(train_loader):
        i = i.to(device)
        j = j.to(device)
        score = model1(i)
        loss = criterion(score,j)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('loss:', loss)


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

check_accuracy(train_loader, model1)