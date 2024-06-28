##!
import torch
##!
#============================================================#
#               initialization Tensor
#============================================================#
device='cuda' if torch.cuda.is_available() else 'cpu'
my_tensor= torch.tensor([[1,2,3],[3,4,5]],dtype=torch.float32, device= device,
                        requires_grad=True) #device='cpu is by default
# print(my_tensor)
# print(my_tensor.dtype,my_tensor.shape)

    #other common initialization

x=torch.empty(size=(3,3)) #creates empty matrix, memory holds random values
x1=torch.zeros(size=(3,3))
x1=torch.ones((3,3))
x1=torch.eye(3,3)
# print(x1)
x2=torch.arange(start=0,end=10,step=1)
# print(x2)
x3=torch.linspace(start=0.1,end=1, steps=10)
# print(x3)

x4=torch.empty((3,3)).normal_(mean=0,std=1)
x41=torch.empty((3,3)).uniform_(0,1)
# print(x41)

x5=torch.diag(torch.ones(3)) #it also creates diagonal matrix, it can be used to any matrix unlike eye
##!
#how to initialize and convert tensors into different type

tensor= torch.arange(4)
'''now converting this tensor into boolian '''
# print(tensor.bool()) #0= False, else= True
# print(tensor.short()) #it converts int16
# print(tensor.long()) #int64
# print(tensor.half()) # float16
# print(tensor.float()) # float32
# print(tensor.double()) #float32

##!
    # array to tensor vice-versa

import numpy as np

np_array= np.zeros((5,5))
tensor1= torch.from_numpy((np_array))
np_array_back= tensor1.numpy()