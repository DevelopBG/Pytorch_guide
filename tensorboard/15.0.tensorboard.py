'''' normal introduction of tensorboard'''
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  #


device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class my_cnn( nn.Module):
    def __init__(self,in_channels,num_classes):
        super().__init__()
        self.conv1= nn.Conv2d(in_channels= in_channels, out_channels= 8, kernel_size=(3,3),
                              stride=(1,1),padding=(1,1))
        self.mpool= nn.MaxPool2d(kernel_size=(2,2),stride= (2,2))
        self.conv2= nn.Conv2d( in_channels=8, out_channels=16, kernel_size=(3,3),
                               stride=(1,1),padding=(1,1))
        self.fc= nn.Linear(16*7*7,num_classes)

    def forward(self,x):
        x= F.relu(self.conv1(x))
        x= self.mpool(x)
        x= F.relu(self.conv2(x))
        x= self.mpool(x)
        x= x.reshape(x.shape[0],-1)
        x= self.fc(x)

        return x

# hyperparameter
batch_size= 64
learning_rate=0.01
num_epoch=5
in_channels=1
num_classes=10
train_dataset= datasets.MNIST(root='datasets/',transform= transforms.ToTensor(),train= True, download= False)
train_loader= DataLoader(train_dataset, batch_size= batch_size, shuffle= True)
test_dataset= datasets.MNIST( root="datasets/", transform= transforms.ToTensor(), train= False,download= True)
test_loader= DataLoader( test_dataset,shuffle= True,batch_size= batch_size)

# optimizer
''''to see tensorboard output , open anaconda prompt, then change environment, change directory to runs,
then >> "tensorboard --logdir runs'"  '''
model= my_cnn(in_channels=in_channels, num_classes=num_classes).to(device)
model.train()
criterion= nn.CrossEntropyLoss()
optimizer= optim.Adam(model.parameters(), lr= learning_rate )
writer= SummaryWriter(f'runs/MNIST/tryingout_tensorboard/epoch{num_epoch}') # address to save the logs


step=0
for epoch in range(num_epoch):
    losses=[]
    accuracies=[]
    for idx,( data, target) in enumerate(train_loader):
        data= data.to(device)
        target= target.to(device)
        scores= model(data)
        loss= criterion(scores, target)
        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred= scores.max(1)
        num_correct = (pred==target).sum()
        training_accuracy= float(num_correct)/float(data.shape[0])
        writer.add_scalar('training loss' , loss, global_step= step)
        writer.add_scalar('training accuracy', training_accuracy, global_step= step)
        step+=1 # step increments x axis , if step =0 a, then all point will be over lapped at point zero






