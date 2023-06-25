import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import *
from utils import *

class CNNNet(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.name = "cnn"
    self.conv1 = nn.Conv2d(1, 8, 3, 1)
    self.conv2 = nn.Conv2d(8, 4, 3, 1)
    self.fc1 = nn.Linear(4 * 16 * 3 * 3, 32)
    self.fc2 = nn.Linear(32, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = Q(x)
    # x = F.relu(x)
    # x = Q(x)
    x = self.conv2(x)
    x = Q(x)
    print('after conv', x.shape)
    # x = F.relu(x)
    # x = Q(x)
    x = F.max_pool2d(x, 2)
    print('after max pool', x.shape)
    x = Q(x)
    x = torch.flatten(x, 1)
    print('after flatten', x.shape)
    x = Q(x)
    x = self.fc1(x)
    x = Q(x)
    # x = F.relu(x)
    # x = Q(x)
    x = self.fc2(x)
    x = Q(x)
    output = F.log_softmax(x, dim=1)
    print('forward done', output)
    return output
  
  def q_forward(self, x):
    def calc(d, n=Q_BITS):
      return ((d * (2 ** n)).to(torch.__dict__[Q_TYPE]).to(torch.float32)) / (2 ** n)
    x = calc(x)
    x = self.conv1(x)
    x = calc(x)
    x = F.relu(x)
    x = calc(x)
    x = self.conv2(x)
    x = calc(x)
    x = F.relu(x)
    x = calc(x)
    x = F.max_pool2d(x, 2)
    x = calc(x)
    x = torch.flatten(x, 1)
    x = calc(x)
    x = self.fc1(x)
    x = calc(x)
    x = F.relu(x)
    x = calc(x)
    x = self.fc2(x)
    x = calc(x)
    output = F.log_softmax(x, dim=1)
    return output

  def manual_forward(self, x):
    if len(x.shape) >= 4:
      return torch.tensor(np.array([self.manual_forward(x[i]).detach().numpy() for i in range(x.shape[0])]))
    x = manual_conv_layer(self.conv1, x)
    x[x < 0] = 0
    x = manual_conv_layer(self.conv2, x)
    x[x < 0] = 0
    x = torch.tensor(fix2floatv(x, Q_BITS), dtype=torch.float32)
    x = F.max_pool2d(x, 2)
    x = Q(x)
    x = torch.flatten(x)
    x = Q(x)
    x = manual_fc_layer(self.fc1, x)
    x[x < 0] = 0
    x = manual_fc_layer(self.fc2, x)
    x = torch.tensor(fix2floatv(x, Q_BITS), dtype=torch.float32)
    output = F.log_softmax(x, dim=0)
    return output