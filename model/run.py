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


def run_model(model):
  summary(model, (1, 28, 28))
  for epoch in range(epochs):
    train(model, device, get_optimizer(model), epoch)
    test(model, device)

def fc():
  model = FcNet().to(device).to(torch.float32)
  run_model(model)
  data = model.state_dict()
  for k in data:
    print(model.name, k, data[k].shape)
    save_path = DATA_PATH + "/" + model.name + '-' + k + ".bin"
    array = data[k].numpy()
    array.astype(np.float32).tofile(save_path)

def cnn():
  model = CNNNet().to(device)
  run_model(model)

if __name__ == '__main__':
  fc()
  # cnn()