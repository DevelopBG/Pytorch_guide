import numpy as np
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!

# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        if in_channels==1:
            self.fc1 = nn.Linear(16 * 7* 8, num_classes)
        else:
            self.fc1 = nn.Linear(16 * 8 * 8, num_classes)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


##........................saving model.................
# def save_model(model,name):
#     print('saving model==>')
#     return torch.save(model.state_dict(),name)
#
# save_model(model,'D:\PYTHON_PROJECTS/backdoor_attack/backdoortrigger_ny/model_backdoor.pth.tar')
###........................................................

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 0.001
batch_size = 200
num_epochs = 15

# Load Data
train_dataset = datasets.CIFAR10(root="datasets/", train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root="datasets/", train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
def training( model, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step
            optimizer.step()
        print(loss.item())

training(model=model,optimizer=optimizer,criterion=criterion,num_epochs=100)

#.................loading model.................................
# my_model = model.Net().to(device)
# pretrained_model = 'D:\PYTHON_PROJECTS/backdoor_attack/backdoortrigger_ny/model_backdoor.pth.tar'
# my_model.load_state_dict(torch.load(pretrained_model, map_location='cpu'))
##....................................................................


# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
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
    # model.train()
    return num_correct/num_samples


# print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
# print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")

###different way to find accuracy....................................
# from sklearn.metrics import accuracy_score
# data_loader= test_loader
# def accuracy(model,data_loader):
#     model.eval()  # switch to eval status
#     y_true = []
#     y_predict = []
#     for step, (batch_x, batch_y) in enumerate(data_loader):
#         batch_y_predict = model(batch_x)
#         batch_y_predict = torch.argmax(batch_y_predict, dim=1)
#         y_predict.append(batch_y_predict)
#         y_true.append(batch_y)
#     y_true = torch.cat(y_true, 0)
#     y_predict = torch.cat(y_predict, 0)
#     print(accuracy_score(y_true.cpu(), y_predict.cpu()))
#
# accuracy(model, test_loader)

