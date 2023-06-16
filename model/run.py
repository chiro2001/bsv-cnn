import torch
import torch.nn.functional as F
from torchsummary import summary
import random
import os

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
def float2fix(float_num: float, decimal_bit: int) -> int:
  fix_num = int(float_num * (2 ** decimal_bit))
  return fix_num

def fix2float(fix_num: int, decimal_bit: int) -> float:
  float_num = fix_num * 1.0 / (2 ** decimal_bit)
  return float_num

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True  # cudnn
  np.random.seed(seed)  # numpy
  random.seed(seed)  # random and transforms

def write_hex(data, path):
  with open(path, "w") as f:
    for i in range(len(data)):
      f.write("{:02X}\n".format(data[i]))

def dump_bsv(model):
  name = model.name.lower()
  with open(GEN_PATH + name + ".bsv", "w") as f:
    f.write(f"// generated file\npackage {name};\n\nimport Vector::*;\n\n")
    def get_layer_name(original: str) -> str:
      return (name + "_" + original).replace(".", "_").replace("-", "_")
    for key in model.state_dict():
      data = model.state_dict()[key]
      data = np.array([float2fix(x, 6) for x in data.flatten()], dtype="int8").reshape(data.shape)
      f.write(f"// model {name} key {key} \n")
      
      def write_function_header(typ: str):
        f.write(f"function {typ} {get_layer_name(key)}();\n")
        f.write(f"  {typ} a;\n")
        
      if len(data.shape) == 1:
        typ = f"Vector#({data.shape[0]}, Int#(8))"
        write_function_header(typ)
        for i in range(data.shape[0]):
          f.write(f"a[{i}]={data[i]};")
        f.write("\n")
        f.write(f"  return a;\n")
        f.write(f"endfunction\n\n")
      elif len(data.shape) == 2:
        typ = f"Vector#({data.shape[0]}, Vector#({data.shape[1]}, Int#(8)))"
        write_function_header(typ)
        for i in range(data.shape[0]):
          for j in range(data.shape[1]):
            f.write(f"a[{i}][{j}]={data[i][j]};")
          f.write("\n")
        f.write(f"  return a;\n")
        f.write(f"endfunction\n\n")
      elif len(data.shape) == 4:
        typ = f"Vector#({data.shape[0]}, Vector#({data.shape[1]}, Vector#({data.shape[2]}, Vector#({data.shape[3]}, Int#(8)))))"
        write_function_header(typ)
        for i in range(data.shape[0]):
          for j in range(data.shape[1]):
            for k in range(data.shape[2]):
              for l in range(data.shape[3]):
                f.write(f"a[{i}][{j}][{k}][{l}]={data[i][j][k][l]};")
            f.write("\n")
        f.write(f"  return a;\n")
        f.write(f"endfunction\n\n")
    
    f.write("endpackage\n")

def dump_binary_hex(model):
  data = model.state_dict()
  data_new = {}
  for k in data:
    save_path = DATA_PATH + model.name + '-' + k + ".bin"
    array = data[k].numpy()
    array.astype(np.float32).tofile(save_path)
    save_path_int8 = DATA_PATH + model.name + '-' + k + ".int8"
    array_uint8 = np.array([float2fix(x, 6) for x in array.flatten()], dtype="uint8")
    # array_int8.tofile(save_path_int8)
    write_hex(array_uint8, save_path_int8)
    print(model.name, k, data[k].shape, "max", array.max(), "min", array.min(), "int8 max", array_uint8.max(), "int8 min", array_uint8.min(), save_path)
    array_restore = np.array([fix2float(x, 6) for x in np.array(array_uint8, dtype="int8").flatten()]).astype(np.float32).reshape(array.shape)
    data_new[k] = torch.from_numpy(array_restore)
    if len(array.shape) >= 2:
      # save as splited lines
      save_path_dir = DATA_PATH + model.name + '-' + k
      if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
      for i in range(array.shape[0]):
        if len(array.shape) == 2:
          a = array_uint8.reshape(array.shape)[i]
        else:
          a = array_uint8.reshape([*list(array.shape)[:-2], array.shape[-1] * array.shape[-2]])[i]
        d = np.hstack(a)
        # print(d)
        # .tofile(save_path_dir + "/" + str(i))
        write_hex(d, save_path_dir + "/" + str(i))
  model.load_state_dict(data_new)
  test(model, device)
        

def run_model(model):
  set_seed(RANDOM_SEED)
  summary(model, (1, 28, 28))
  for epoch in range(epochs):
    train(model, device, get_optimizer(model), epoch)
    test(model, device)
  dump_bsv(model)

def fc():
  model = FcNet().to(device).to(torch.float32)
  run_model(model)

def cnn():
  model = CNNNet().to(device)
  run_model(model)

def test_floats():
  num = 127.2133241
  num2 = -2.3
  decimal_bit: int = 8
  fix = float2fix(num, decimal_bit)
  fix2 = float2fix(num2, decimal_bit)
  fl = fix2float(fix, decimal_bit)
  fl2 = fix2float(fix2, decimal_bit)
  f = fix2float(fix * fix2, decimal_bit * 2)
  print(num, fix, fl)
  print(num2, fix, fl2)
  print(num * num2, f, fl * fl2)

if __name__ == '__main__':
  fc()
  cnn()