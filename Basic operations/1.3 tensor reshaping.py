import torch

#======================================================================
#                           Tensor Reshaping
#======================================================================

x=torch.arange(9)
x_3x3= x.view(3,3)
x_3x3= x.reshape(3,3)  #''' view and reshape nearly same, there are some difference you can found in documentaion.
                            #preferable to use view for better performance  but for simplicity use reshape'''

y= x_3x3.t() #transpose
# print(x_3x3,y)

x1= torch.randn((3,3))
x2=torch.randn((3,3))
y1= torch.cat((x1,x2),dim=1)
# print(y1)

z3= x1.view(-1) #flatten the tensor
# print(z3.shape)

batch= 64
x5= torch.randn((batch,3,4))
z4= x5.view(batch,-1)
# print(z4.shape)

z5= x5.permute( 0,2,1) # batch will be same but dimension 2nd will be 1st and 1 st will be 2nd
# print('original shape: {} \n and\n modified shape: {}'. format(x5.shape,z5.shape))


x6= torch.arange(10)
z6=x6.unsqueeze(0)
print(z6)
z7=x6.unsqueeze(1)
print(z7)
x7= torch.arange(10).unsqueeze(0).unsqueeze(1) #1x1x10

z8=x7.squeeze(1) #1x10, omitting first dimension