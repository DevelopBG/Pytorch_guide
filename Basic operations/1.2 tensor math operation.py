##!
import torch
##!
#========================================================================
#               Tensor Math & Comparison Operations
#========================================================================
    #add
x1=torch.tensor([1,2,3])
y=torch.tensor([2,3,4])
z1=torch.empty(3)
torch.add(x1,y,out=z1)
    #  or
torch.add(x1,z1)
    # division
z=torch.true_divide(x,y) #element wise division but if y is integer then every element will be divided by y

    # inplace operation
'''this mutate tensor , which motivate not to copy
tensors.inplace done using "_" at the end of the operation, it reduce computational burden'''
t= torch.zeros(3)
# print(t)
t.add_(x1)
# or
t+=x1 # but t=t+x does not work inplace , this will create copy
# print(t)

    # exponentiation
z2= x1.pow(2) # element wise square
# or
z2= x1**2

    # element wise comparison
z2= x1>0
# print(z2)

    #matrix multiplication

x1=torch.rand((2,5))
x2=torch.rand((5,3))
x3=torch.mm(x1,x2)
    #or
x3=x1.mm(x2)
##!
    # matrix exponentiation
''' don't wanna element wise exponentiation,rather multiplying matrix by itself n times, only square matrix 
 will be valid'''
mat1= torch.tensor([[1,2],[3,4]])
mat2=torch.tensor([[2,3],[4,3]])
# print(mat1.pow(3))
# print(mat1.matrix_power(3)) #it is actually mat1*mat1*mat1
z=mat1**mat2# each element of mat1 will powered corresponding mat2 elements e.g. 2**3,4**3...
# print(z)
##!
    #element wise multiplication
p=torch.tensor([1,2,3])
p1=torch.tensor([2,3,4])
z=p*p1
# print(z)
    #dot product
z=torch.dot(p,p1)
##!
    #batch matrix multiplication

batch=32
n=10
m=20
p=30

tensor1= torch.rand((batch,n,m))
tensor2= torch.rand((batch,m,p))
out= torch.bmm(tensor1,tensor2) # (batch,n,p)
print(out.shape)

#   Example of broadcasting

q= torch.rand((5,5))
q1=torch.randn((1,5))
z=q-q1
##!
# other useful tensor operation
x1=torch.tensor([[1,2,3,4],[3,4,5,6]])
sum_x=torch.sum(x, dim=0)
# print(sum_x)
values, indices= torch.max(x,dim=0) # returns max valued row or column
# print(values,indices)
z=torch.abs(x1) #returns absolute value
# print(z)
x=torch.tensor([2,3,4,5,6])
z1=torch.argmax(x,dim=0)
z1=torch.argmin(x,dim=0)
print(z1)
mean_x= torch.mean(x1.float(),dim=0) # x1.mean(dim=0) #to compute mean, tensor element needs to be float
sorted_x1,indices=torch.sort(x1, dim=0, descending=True)
print(indices,sorted_x1)

z=torch.clamp(x1,min=2,max=3) #if any element is less than 2 then it will be fixed to 2 and any greater than 3 , will be set at 3
print(z)
x=torch.tensor([1,2,0,0,3],dtype= torch.bool)
z= torch.any(x) # if any is greater than zero, it will return true
print(z)
z1= torch.all(x) #all elements need to be greater than zero, then it will return true else false
print(z1)
