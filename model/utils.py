import numpy as np
from config import *

# 浮点数转换为定点数
def float2fix(float_num: float, decimal_bit: int) -> int:
  fix_num = int(float_num * (2 ** decimal_bit))
  return fix_num

# 定点数转换为浮点数
def fix2float(fix_num: int, decimal_bit: int) -> float:
  float_num = fix_num * 1.0 / (2 ** decimal_bit)
  return float_num

def down_precision(num: float, decimal_bit: int) -> float:
  return fix2float(float2fix(num, decimal_bit), decimal_bit)

def float2fixv(float_num, decimal_bit: int, dtype=np.int32):
  fix_num = (np.array(float_num) * (2 ** decimal_bit)).astype(dtype)
  return fix_num

def fix2floatv(fix_num, decimal_bit: int):
  float_num = np.array(fix_num).astype(np.float32) / (2 ** decimal_bit)
  return float_num

def manual_fc_layer(layer, x):
  dtype = np.__dict__[Q_TYPE]
  valid_bits = int(Q_TYPE[3:])
  dtype2 = np.__dict__[f'int{valid_bits * 2}']
  x0 = x if isinstance(x, np.ndarray) else float2fixv(x.clone().detach().numpy(), Q_BITS, dtype=dtype)
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
  return r0

def manual_conv_layer(layer, x):
  dtype = np.__dict__[Q_TYPE]
  valid_bits = int(Q_TYPE[3:])
  dtype2 = np.__dict__[f'int{valid_bits * 2}']
  x0 = x if isinstance(x, np.ndarray) else float2fixv(x.clone().detach().numpy(), Q_BITS, dtype=dtype)
  if len(x0.shape) == 3:
    x0 = x0.reshape((1, *x0.shape))
  w = float2fixv(layer.weight.clone().detach().numpy(), Q_BITS, dtype=dtype)
  b = float2fixv(layer.bias.clone().detach().numpy(), Q_BITS, dtype=dtype)
  kernel_size = 3
  out_size = (x0.shape[2] + 1 - kernel_size, x0.shape[3] + 1 - kernel_size)
  output_channel = w.shape[0]
  out = np.zeros((output_channel, *out_size), dtype=dtype2)
  print(f"Conv: {x0.shape}, {w.shape}, {b.shape} -> {out.shape}")
  for c in range(output_channel):
    for i in range(out_size[0]):
      for j in range(out_size[1]):
        xx = x0[0, :, i:i+kernel_size, j:j+kernel_size].flatten()
        ww = w[c, :].flatten()
        rr = dtype2(0)
        for xi in range(len(xx)):
          mul_raw = dtype2(dtype2(xx[xi]) * dtype2(ww[xi]))
          mul = dtype2(mul_raw >> Q_BITS)
          rr = dtype2(rr + mul)
        rr = dtype(rr)
        out[c, i, j] = rr + b[c]
  return out