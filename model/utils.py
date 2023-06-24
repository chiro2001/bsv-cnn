import numpy as np

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