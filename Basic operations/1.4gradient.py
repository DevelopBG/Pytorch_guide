import torch

## gradinet when output is scalar

x = torch.rand(3, requires_grad= True)
y = 2*x
z = y**2 +1
z = z.mean()
z.backward()
print(x.grad)

## when output is a venctor :

x1 = torch.tensor([2.,1.,3.], requires_grad= True)
p1 = torch.tensor([1.,3.,4.], requires_grad= True)
y1 = x1**2 + p1**2 + x1
z1 = y1
vector = torch.tensor([1.,1.,1.], dtype = torch.float32)
z1.backward(vector)
print(x1.grad) # x1.grad = dz1/x1 ; p1.grad = dz1/p1

##.... torch loss function...

import torch
import torch.nn as nn
from torch import linalg as LA

loss = nn.CrossEntropyLoss()
loss1 = nn.MSELoss()

# if you will replace the dtype=torch.float, you will get error

input = torch.randn((1,3, 5),dtype= torch.float32, requires_grad=True)
target = torch.ones((1,5), dtype=torch.long)
print(target)
x = LA.norm(input)
output = loss(input, target)
output.backward()
print(input.grad)
####.......
