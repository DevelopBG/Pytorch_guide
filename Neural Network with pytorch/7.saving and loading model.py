import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader # DataLoader gives us many import activities, like mini-batch, tensor etc
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from tqdm import tqdm

        # data loading
train_dataset= datasets.MNIST(root='datasets/',train= True, transform= transforms.ToTensor(), download= True)
train_loader= DataLoader( dataset= train_dataset,batch_size= 64, shuffle= True)
test_dataset= datasets.MNIST(root='datasets/',transform= transforms.ToTensor(), train=False, download= True)
test_loader= DataLoader( dataset= test_dataset, shuffle= True, batch_size=64)
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # simple CNN
class CNN1(nn.Module):
    def __init__(self,inp,num_classes):
        super().__init__()
        self.conv1= nn.Conv2d(in_channels= inp, out_channels= 10, kernel_size= (3,3),stride= (1,1),padding= (1,1))
        self.pool= nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2= nn.Conv2d(in_channels= 10, out_channels=16, kernel_size= (3,3), stride= (1,1), padding= (1,1))
        self.fc= nn.Linear(16*7*7,num_classes)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x= self.pool(x)
        x= F.relu(self.conv2(x))
        x= self.pool(x)
        x= x.reshape(x.shape[0],-1)
        x= self.fc(x)

        return x

        # model saving
def save_checkpoint(state,filename="my_checkpoint.pth.tar"):
    print('=>saving checkpoint')
    torch.save(state,filename)

        # model loading function
def load_checkpoint(checkpoint):
    print('loading checkpoint \n')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

device= torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
epochs=5
batch_size=64
learning_rate= 0.001
inp= 1
load_model=True # model loading hyperparameter, if false, it will not load

model= CNN1(inp=inp,num_classes=10).to(device=device )
optimizer= optim.Adam(model.parameters(),lr= learning_rate)
criterion= nn.CrossEntropyLoss()

        # loading model
if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))

        # training
for epoch in range(epochs):
    losses=[]
    if epoch%2==0:
        # saving model
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}           ## saving
        save_checkpoint(checkpoint)

    for idx, (data,label) in enumerate(train_loader):
        data= data.to(device=device)
        label= label.to(device=device)

        score= model(data)
        loss= criterion( score, label)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    mean_loss= sum(losses)/len(losses)
    print(f'loss at epoch {epoch} is {mean_loss:.4f}')



def check_accuracy(loader, model):
    if loader.dataset.train:
        print('accuracy for training set')
    else:
        print('accuracy for testing')

    num_correct= 0
    num_samples=0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x= x.to(device=device)
            y= y.to(device=device)

            scores= model(x)
            _, pred= scores.max(1)
            num_correct+= (pred==y).sum()
            num_samples += pred.size(0)

        print(f'accuracy = {float(num_correct)/float(num_samples)}')

    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)



