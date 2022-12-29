import torch

w = torch.tensor( [[1.,2.],[4.,5.]], requires_grad=True)
function = 10 * torch.log( ( w + 1. ) ).sum()
function.backward()
print(w-w.grad)