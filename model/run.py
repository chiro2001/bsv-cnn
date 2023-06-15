import torch
import torch.nn.functional as F
from torchsummary import summary
from data_helper import *
from config import *

from fc import FcNet
from cnn import CNNNet

def train(model, device, optimizer, epoch, log_interval=100):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))


def test(model, device):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# 浮点数转换为定点数
def float2fix(float_num: float, decimal_bit: int, fix_bit: int) -> int:
  # float_num 输入浮点数
  # decimal_bit 小数位位数
  # fix_bit 定点数位数
  # fix_num = (2 ** fix_bit + np.round(float_num * (2 ** decimal_bit))) % (2 ** fix_bit)
  # fix_num = 2 ** fix_bit + np.round(float_num * (2 ** decimal_bit))
  # fix_num - 2 ** fix_bit = float_num * (2 ** decimal_bit)
  # fix_num / (2 ** decimal_bit) - 2 ** (fix_bit - decimal_bit) = float_num
  fix_num = int(float_num * (2 ** decimal_bit))
  return fix_num

def fix2float(fix_num: int, decimal_bit: int, fix_bit: int) -> float:
  # fix_num 输入定点数
  # decimal_bit 小数位位数
  # fix_bit 定点数位数
  # float_num = fix_num / (2 ** decimal_bit) - 2 ** (fix_bit - decimal_bit)
  float_num = fix_num * 1.0 / (2 ** decimal_bit)
  return float_num

def run_model(model):
  summary(model, (1, 28, 28))
  for epoch in range(epochs):
    train(model, device, get_optimizer(model), epoch)
    test(model, device)
  data = model.state_dict()
  for k in data:
    save_path = DATA_PATH + model.name + '-' + k + ".bin"
    array = data[k].numpy()
    array.astype(np.float32).tofile(save_path)
    save_path_int8 = DATA_PATH + model.name + '-' + k + ".int8"
    array_int8 = np.array([float2fix(x, 8, 6) for x in array.flatten()])
    array_int8.astype(np.int8).tofile(save_path_int8)
    print(model.name, k, data[k].shape, "max", array.max(), "min", array.min(), "int8 max", array_int8.max(), "int8 min", array_int8.min(), save_path)

def fc():
  model = FcNet().to(device).to(torch.float32)
  run_model(model)

def cnn():
  model = CNNNet().to(device).to(torch.float32)
  run_model(model)

if __name__ == '__main__':
  num = -89
  decimal_bit: int = 8
  fix_bit: int = 3
  fix = float2fix(num, decimal_bit, fix_bit)
  print(num, fix, fix2float(fix, decimal_bit, fix_bit))
  fc()
  # cnn()