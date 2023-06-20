import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *

class FcNet(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.name = "fc"
    self.fc1 = nn.Linear(784, 32)
    self.fc2 = nn.Linear(32, 10)

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    # x = F.relu(x)
    x = self.fc2(x)
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

