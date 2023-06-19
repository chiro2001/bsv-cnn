import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNNet(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.name = "cnn"
    self.conv1 = nn.Conv2d(1, 8, 3, 1)
    self.conv2 = nn.Conv2d(8, 4, 3, 1)
    self.fc1 = nn.Linear(4*16*3*3, 32)
    self.fc2 = nn.Linear(32, 10)

  def forward(self, x):
    # print("data", x.shape)
    x = self.conv1(x)
    # print("conv1", x.shape)
    x = F.relu(x)
    x = self.conv2(x)
    # print("conv2", x.shape)
    x = F.relu(x)
    x = F.max_pool2d(x, 2)
    # print("max_pool2d", x.shape)
    x = torch.flatten(x, 1)
    # print("flatten", x.shape)
    x = self.fc1(x)
    # print("fc1", x.shape)
    x = F.relu(x)
    x = self.fc2(x)
    # print("fc2", x.shape)
    output = F.log_softmax(x, dim=1)
    return output

