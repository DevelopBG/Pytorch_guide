import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision.datasets as datasets

my_transforms= transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64,64)),
    transforms.RandomCrop((62,62)),
    transforms.ColorJitter(brightness= 0.5),
    transforms.RandomHorizontalFlip(p=0.5), # p= probability
    transforms.RandomRotation(degrees=45),
    transforms.RandomVerticalFlip(p= 0.05),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.2,0.2,0.1],std=[1.,1.,1.]) # (value-mean)/ std
])

dataset= datasets.MNIST( root='datasets/', transform= my_transforms, download= True)

