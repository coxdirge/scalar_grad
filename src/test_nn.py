import torch
from engine import Value

# micrograd
a = Value(2.0)
b = Value(-3.0)

y = (a * b + a).tanh()
y.backward()

# PyTorch
ta = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
tb = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)

ty = torch.tanh(ta * tb + ta)
ty.backward()

print("micrograd:", a.grad, b.grad)
print("PyTorch:  ", ta.grad.item(), tb.grad.item())

assert abs(a.grad - ta.grad.item()) < 1e-9
assert abs(b.grad - tb.grad.item()) < 1e-9
