import torch
x= torch.tensor([3,4,5])
torch.save(x,'tensor.pt')
print(torch.load('tensor.pt'))



x= torch.randn((3,2))
y= torch.tensor([3,4,5])
z=zip(x,y)
torch.save(z,'tensor.pt')
p=torch.load('tensor.pt')
for i,j in p:
    print(i,j)

torch.cuda.empty_cache() # cache memory clearing
