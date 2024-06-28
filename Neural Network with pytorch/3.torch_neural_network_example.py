#line up of the work .py
# imports-> create fully connected network-> hyperparameter->load data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader # DataLoader gives usmany import activities, like minibatch, tensor etc
import torchvision.datasets as datasets
import torchvision.transforms as transforms

            # creating a simple NN model
class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        super().__init__()
        self.fc1= nn.Linear(input_size, 50)
        self.fc2=nn.Linear(50,num_classes)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)

        return x

                # just of testing the model, not required

# model=NN(120,5)
# x=torch.randn(64, 120)  # 64= no of example and input size; 64 no of examples will running simulataneously
# print(model(x).shape) # out= (64,5) , means each samples gets its class

                # set device for GPU or CPU

device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

                # hyperparameters
input_size= 784
learning_rate= 0.001
num_classes= 10
batch_size= 64
epochs= 1

                # load data
train_dataset= datasets.MNIST(root='datasets/',train= True, transform= transforms.ToTensor(), download= True)
train_loader= DataLoader( dataset= train_dataset,batch_size= batch_size, shuffle= True)
test_dataset= datasets.MNIST(root='dataset/',train= False, transform= transforms.ToTensor(),download= True)
test_loader= DataLoader( dataset= test_dataset, batch_size= batch_size,shuffle= True)

                # model initialization
model= NN( input_size= input_size, num_classes= num_classes).to(device)

                # loss and optimization
criterion= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(),lr= learning_rate)

                # Train network

for epoch in range(epochs):
    for batch_idx , (data, target) in enumerate(train_loader):
        # getting data to cuda
        data=data.to(device=device)
        target= target.to(device=device)
        # print(data.shape)

        # get to correct shape
        data= data.reshape(data.shape[0],-1)

        # forward
        scores= model(data)
        loss= criterion(scores, target)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient and parameter update
        optimizer.step()

# checking accuracy on training and testing to see the model's performance
def check_accuracy(loader, model):

    if loader.dataset.train:
        print(' checking training accuracy ')
    else:
        print(' checking testing accuracy')
    num_correct=0
    num_sample=0
    model.eval() # let the model know it is an evaluation mode

    with torch.no_grad(): # when we check the accuracy do not need gradient
        for x,y in loader:
            x= x.to(device=device)
            y=y.to(device=device)
            x= x.reshape(x.shape[0],-1)

            scores= model(x) # 64x10
            _, prediction= scores.max(1) # as we need to find the index no of second dim of scores

            num_correct += ( prediction == y).sum()
            num_sample += prediction.size(0)
        acc=float(num_correct)/float(num_sample)*100
        print(f' accuracy = {float(num_correct)/float(num_sample):.2f}') # converting tensors into float

    # model.train()
    return acc

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)






