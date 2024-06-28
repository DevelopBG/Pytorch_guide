''' here we will find different batch_size effect and learning_rate effect ond
 accuracy and losses'''
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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

num_epoch=1
in_channels=1
num_classes=10
train_dataset= datasets.MNIST(root='datasets/',transform= transforms.ToTensor(),train= True, download= False)

test_dataset= datasets.MNIST( root="datasets/", transform= transforms.ToTensor(), train= False,download= True)
# test_loader= DataLoader( test_dataset,shuffle= True,batch_size= batch_size)

# optimizer


step=0
batch_sizes=[64,1028]
learning_rates=[0.1,0.01,0.001]
for batch_size in batch_sizes:
    for learning_rate in learning_rates:
        model = my_cnn(in_channels=in_channels, num_classes=num_classes).to(device)
        model.train()
        writer = SummaryWriter(f'runs/MNIST/miniBatchSize {batch_size} LR {learning_rate}')  # address to save the logs
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.0)
        for epoch in tqdm(range(num_epoch)):
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
                # calculate running accuracy

                training_accuracy= float(num_correct)/float(data.shape[0])
                accuracies.append(training_accuracy)
                writer.add_scalar('training loss' , loss, global_step= step)
                writer.add_scalar('training accuracy', training_accuracy, global_step= step)
                step+=1 # step increments x axis , if step =0 a, then all point will be over lapped at point zero

        writer.add_hparams({'lr':learning_rate, 'bsize': batch_size},
                           {'accuracy': sum(accuracies)/len(accuracies),
                            'loss': sum(losses)/len(losses)})




