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
    self.fc1 = nn.Linear(784, 64)
    self.fc2 = nn.Linear(64, 10)

  def forward(self, x):
    x = torch.flatten(x, 1)
    x = Q(x)
    x = self.fc1(x)
    x = Q(x)
    x = F.relu(x)
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
    x = F.relu(x)
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
    x = torch.flatten(x)
    x = manual_fc_layer(self.fc1, x)
    # Relu
    x[x < 0] = 0
    x = manual_fc_layer(self.fc2, x)
    x = torch.tensor(fix2floatv(x, Q_BITS), dtype=torch.float32)
    output = F.log_softmax(x, dim=0)
    return output

