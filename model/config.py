import torch

DATA_PATH = "../data/"
GEN_PATH = "../gen/"
RANDOM_SEED = 114514
batch_size_train = 64
batch_size_test = 1000
LEARNING_RATE = 0.01

# Q_TYPE = "int16"
# Q_BITS = 8

# Q_TYPE = "int16"
# Q_BITS = 9

# Q_TYPE = "int8"
# Q_BITS = 2

Q_TYPE = "int32"
Q_BITS = 20

# Q_TYPE = "int32"
# Q_BITS = 12

MODEL_CACHE = True

device = torch.device("cpu")
epochs = 1

def get_optimizer(model):
  return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# First define number formats used in forward and backward quantization
from qtorch import FixedPoint, FloatingPoint
forward_num = FixedPoint(wl=int(Q_TYPE[3:]), fl=Q_BITS, clamp=True)
# backward_num = FloatingPoint(exp=8, man=5)
backward_num = forward_num
# backward_num = FixedPoint(wl=32, fl=20)

# Create a quantizer
from qtorch.quant import Quantizer
Q = Quantizer(forward_number=forward_num, backward_number=backward_num,
              forward_rounding="nearest", backward_rounding="stochastic")