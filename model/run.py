import torch
import torch.nn.functional as F
from torchsummary import summary
import random
import os

from utils import *
from data_helper import *
from config import *
from fc import FcNet
from cnn import CNNNet

def train(model, device, optimizer, epoch, log_interval=100):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    # print('data', data.shape, 'mean', data.mean(), 'max', data.max(), 'min', data.min())
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

def q_test(model, device):
  model.eval()
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model.q_forward(data)
      # print('output', output.shape, output)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  print('Test Q: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

def manual_test(model, device):
  model.eval()
  correct = 0
  count = 0
  dataset_use = train_loader
  # dataset_use = test_loader
  with torch.no_grad():
    for data, target in dataset_use:
      data, target = data.to(device), target.to(device)
      output = model.manual_forward(data)
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()
      count += len(output)
      break

  # print('Test Manual: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(dataset_use.dataset), 100. * correct / len(dataset_use.dataset)))
  print('Test Manual: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, count, 100. * correct / count))

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True  # cudnn
  np.random.seed(seed)  # numpy
  random.seed(seed)  # random and transforms

def write_hex(data, path):
  fmt = ("{:0" + str(int(Q_TYPE[3:]) // 4) + "X}\n")
  with open(path, "w") as f:
    for i in range(len(data)):
      d = fmt.format(int(data[i]))
      # print('fmt', fmt[:-1], 'd', d[:-1], 'data', data[i], 'int(data[i])', int(data[i]))
      f.write(d)


# write_hex_2d_g = False

def write_hex_2d(data, path):
  # global write_hex_2d_g
  # print('write_hex_2d: data shape', data.shape)
  fmt = ("{:0" + str(int(Q_TYPE[3:]) // 4) + "X}")
  data = data.reshape([data.shape[0], -1])
  with open(path, "w") as f:
    for i in range(data.shape[0]):
      for j in range(data.shape[1]):
        # f.write(fmt.format(data[i][j]))
        # f.write(fmt.format(data[data.shape[0] - 1 - i][j]))
        f.write(fmt.format(data[i][data.shape[1] - 1 - j]))
        # if not write_hex_2d_g:
        #   print(np.__dict__[Q_TYPE](data[i][j]), end='\t')
      f.write("\n")
      # if not write_hex_2d_g:
      #   print('')
      #   write_hex_2d_g = True
  # if not write_hex_2d_g:
  #   write_hex_2d_g = True

def dump_bsv(model, vector: bool = False):
  name = model.name.lower()
  with open(GEN_PATH + name + ".bsv", "w") as f:
    f.write(f"// generated file\npackage {name};\n\nimport Vector::*;\n\n")
    def get_layer_name(original: str) -> str:
      return (name + "_" + original).replace(".", "_").replace("-", "_")
    for key in model.state_dict():
      data = model.state_dict()[key]
      data = np.array([float2fix(x, 6) for x in data.flatten()], dtype="int8").reshape(data.shape)
      f.write(f"// model {name} key {key} \n")
      
      def write_function_header(typ: str, typ2=None, typ3="a"):
        typ2 = typ2 if typ2 is not None else typ
        f.write(f"function {typ2} {get_layer_name(key)}();\n")
        f.write(f"  {typ} {typ3};\n")
      
      if vector:
        if len(data.shape) == 1:
          typ = f"Vector#({data.shape[0]}, Int#(8))"
          write_function_header(typ)
          for i in range(data.shape[0]):
            f.write(f"a[{i}]={data[i]};")
          f.write("\n")
        elif len(data.shape) == 2:
          typ = f"Vector#({data.shape[0]}, Vector#({data.shape[1]}, Int#(8)))"
          write_function_header(typ)
          for i in range(data.shape[0]):
            for j in range(data.shape[1]):
              f.write(f"a[{i}][{j}]={data[i][j]};")
            f.write("\n")
        elif len(data.shape) == 4:
          typ = f"Vector#({data.shape[0]}, Vector#({data.shape[1]}, Vector#({data.shape[2]}, Vector#({data.shape[3]}, Int#(8)))))"
          write_function_header(typ)
          for i in range(data.shape[0]):
            for j in range(data.shape[1]):
              for k in range(data.shape[2]):
                for l in range(data.shape[3]):
                  f.write(f"a[{i}][{j}][{k}][{l}]={data[i][j][k][l]};")
              f.write("\n")
      else:
        if len(data.shape) == 1:
          write_function_header(f"Int#(8) a[{data.shape[0]}]", "Int#(8)[]", "")
          for i in range(data.shape[0]):
            f.write(f"a[{i}]={data[i]};")
          f.write("\n")
        elif len(data.shape) == 2:
          write_function_header(f"Int#(8) a[{data.shape[0]}][{data.shape[1]}]", "Int#(8)[][]", "")
          for i in range(data.shape[0]):
            for j in range(data.shape[1]):
              f.write(f"a[{i}][{j}]={data[i][j]};")
            f.write("\n")
        elif len(data.shape) == 4:
          write_function_header(f"Int#(8) a[{data.shape[0]}][{data.shape[1]}][{data.shape[2]}][{data.shape[3]}]", "Int#(8)[][][][]", "")
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
  n = Q_BITS
  dtype = Q_TYPE
  for k in data:
    array = Q(data[k]).numpy()
    save_path_hex = DATA_PATH + model.name + '-' + k + ".hex"
    array_uint = np.array([float2fix(x, n) for x in array.flatten()], dtype="u" + dtype)
    if len(array.shape) == 1:
      write_hex(array_uint, save_path_hex)
    elif len(array.shape) == 2:
      write_hex_2d(array_uint.reshape(array.shape).T, save_path_hex)
    else:
      write_hex_2d(array_uint.reshape(array.shape), save_path_hex)
    print(model.name, k, data[k].shape, "max", array.max(), "min", array.min(), dtype, "max", array_uint.max(), dtype, "min", array_uint.min(), save_path_hex)
    array_restore = np.array([fix2float(x, n) for x in np.array(array_uint, dtype=dtype).flatten()]).astype(np.float32).reshape(array.shape)
    data_new[k] = torch.from_numpy(array_restore)
    print('restore diff mean', np.mean(array_restore - array), 'max', np.max(array_restore - array), 'min', np.min(array_restore - array))
  model.load_state_dict(data_new)
  # q_test(model, device)
        

def run_model(model, model_path: str = "", test_manual: bool = False):
  set_seed(RANDOM_SEED)
  summary(model, (1, 28, 28))
  if not MODEL_CACHE or not os.path.exists(model_path):
    for epoch in range(epochs):
      train(model, device, get_optimizer(model), epoch)
      test(model, device)
  # save model
  if len(model_path) > 0:
    print("save model to", model_path)
    torch.save(model.state_dict(), model_path)
  if test_manual:
    manual_test(model, device)
  # dump_bsv(model)
  dump_binary_hex(model)

def fc():
  model = FcNet().to(device).to(torch.float32)
  model_path = "fc.pt"
  if MODEL_CACHE and os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
  run_model(model, model_path=model_path, test_manual=True)

def cnn():
  model = CNNNet().to(device)
  model_path = "cnn.pt"
  if MODEL_CACHE and os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
  run_model(model, model_path=model_path)

def test_floats():
  num = 127.2133241
  num2 = -2.3
  decimal_bit: int = 8
  fix = float2fix(num, decimal_bit)
  fix2 = float2fix(num2, decimal_bit)
  fl = fix2float(fix, decimal_bit)
  fl2 = fix2float(fix2, decimal_bit)
  f = fix2float(fix * fix2, decimal_bit * 2)
  f2 = fix2float((fix * fix2) >> decimal_bit, decimal_bit)
  print(num, fix, fl)
  print(num2, fix, fl2)
  print(num * num2, f, f2, fl * fl2)

def dump_test_set():
  lst = []
  testset_use = train_loader
  for data, target in testset_use:
    data, target = data.to(device), target.to(device)
    lst.append((data, target))
    break
  # random.shuffle(lst)
  data, target = lst[0]
  # data, target = Q(data.to(torch.float32)), Q(target.to(torch.float32))
  data, target = data.to(torch.float32), target.to(torch.float32)
  data, target = data.numpy(), target.numpy()
  data = np.array([float2fix(x, Q_BITS) for x in data.flatten()], dtype="u" + Q_TYPE).reshape(data.shape)
  target = np.array([float2fix(x, Q_BITS) for x in target.flatten()], dtype="u" + Q_TYPE)
  print('data', data.shape, 'target', target.shape)
  path = DATA_PATH + "test_input"
  write_hex_2d(data, path + ".data" + ".hex")
  write_hex(target, path + ".target" + ".hex")

if __name__ == '__main__':
  # test_floats()
  dump_test_set()
  fc()
  # cnn()