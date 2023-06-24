import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *
from utils import *

class FcNet(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.name = "fc"
    self.fc1 = nn.Linear(784, 32)
    self.fc2 = nn.Linear(32, 10)

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = Q(x)
    x = self.fc1(x)
    x = Q(x)
    # x = F.relu(x)
    x = self.fc2(x)
    x = Q(x)
    output = F.log_softmax(x, dim=1)
    return output
  
  def q_forward(self, x):
    def calc(d, n=Q_BITS):
      return ((d * (2 ** n)).to(torch.__dict__[Q_TYPE]).to(torch.float32)) / (2 ** n)
    x = calc(x)
    x = torch.flatten(x, 1)
    x = calc(x)
    x = self.fc1(x)
    x = calc(x)
    x = self.fc2(x)
    x = calc(x)
    output = F.log_softmax(x, dim=1)
    return output

  def manual_forward(self, x):
    if len(x.shape) >= 4:
      return torch.tensor(np.array([self.manual_forward(x[i]).detach().numpy() for i in range(x.shape[0])]))
    def calc_layer(layer, x):
      dtype = np.__dict__[Q_TYPE]
      valid_bits = int(Q_TYPE[-2:])
      dtype2 = np.__dict__[f'int{valid_bits * 2}']
      x0 = float2fixv(x.clone().detach().numpy(), Q_BITS, dtype=dtype)
      w = float2fixv(layer.weight.clone().detach().numpy(), Q_BITS, dtype=dtype)
      b = float2fixv(layer.bias.clone().detach().numpy(), Q_BITS, dtype=dtype)
      r0 = [dtype2(0) for _ in range(len(b))]
      for i in range(len(x0)):
        for j in range(len(b)):
          mul_raw = dtype2(dtype2(x0[i]) * dtype2(w[j][i]))
          mul = dtype2(mul_raw >> Q_BITS)
          r0[j] = dtype2(r0[j] + mul)
          if i == j:
            r0[i] = dtype2(r0[i] + b[i])
      r0 = dtype(r0)
      r0 = fix2floatv(r0, Q_BITS)
      return torch.tensor(r0)
    x = torch.flatten(x)
    x = Q(x)
    # x = torch.matmul(self.fc1.weight, x) + self.fc1.bias
    x = calc_layer(self.fc1, x)
    x = Q(x)
    # x = torch.matmul(self.fc2.weight, x) + self.fc2.bias
    x = calc_layer(self.fc2, x)
    x = Q(x)
    output = F.log_softmax(x, dim=0)
    return output

