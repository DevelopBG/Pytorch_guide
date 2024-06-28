##!
import torch

#============================================================================================
                                  # Tensor Indexing
#============================================================================================
##!
batch_size=10
feature=25
x=torch.rand((batch_size,feature))

print(x[0]) # all features for first example
print(x[:,0]) # all example for first feature

print(x[2,0:10])

x[0,0]=100 # assigning the value to the first element

##!
    #fancy indexing

x1= torch.arange(10)
indices=[2,3,4]
print(x[indices]) # it returns indices element from x1

x2= torch.randn((5,6))
rows= torch.tensor([0,3])
cols= torch.tensor([2,3])
print(x2[rows,cols])

##!
                #more advance Indexing

x3= torch.arange(10)
z0=x3[(x3<2) | (x3>7)] # '|'-> 'or' and '&' -> 'and'
z=x3[x3.remainder(2)==0]

z1=torch.where(x3>5,0,x3*2) # if x> 3, then set the value to '0' else x*2
# print(z1)
x4= torch.tensor([0,0,1,1,2,3,3,4,4,5])
z2=x4.unique()
# print(z2)

print( x4.ndimension()) # if x= shape of 5x5x5 , then it returns 3.

print(x4.numel()) #count the number of elements
