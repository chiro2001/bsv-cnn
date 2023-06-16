import torch
import torch.nn as nn
import torch.nn.functional as F

class FcNet(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.name = "fc"
    self.fc1 = nn.Linear(784, 128)
    self.fc2 = nn.Linear(128, 10)
    d = self.fc1.state_dict()
    # print(d['weight'].shape)
    # print(d['bias'].shape)

  def forward(self, x):
    # print(x.shape)
    x = torch.flatten(x, 1)
    # print(x.shape)
    # print(torch.matmul(self.fc1.state_dict()['weight'], x[0]))
    x = self.fc1(x)
    # print(x.shape)
    x = F.relu(x)
    # print(x.shape)
    x = self.fc2(x)
    # print(x.shape)
    output = F.log_softmax(x, dim=1)
    # print(output.shape)
    # print(output)
    # exit(0)
    return output

