import torch

a = torch.rand(1,3)
print(a.expand(3,3))
print(a.repeat(3,1))