import torch
k = torch.tensor([50000000.00000])
t = torch.tensor(.01)
data_tensor = torch.sign(k)*torch.log10(k)*2.0/45.0
print data_tensor
l = torch.tensor(10.0)**(45.0/2.0*torch.abs(data_tensor))*torch.sign(data_tensor)
lp = torch.tensor(10.0)**(45/2*torch.abs(data_tensor))*torch.sign(data_tensor)
print l
print lp
print k
#l = torch.DoubleTensor([10.0])**(45/2*torch.abs(data_tensor))*t*torch.sign(data_tensor)
