import torch

a = torch.randn((2,2))
print(a.size())
a = torch.reshape(a, (2,2,1))
print(a.size())

