import torch

DATA_PATH = "../data/"
GEN_PATH = "../gen/"
RANDOM_SEED = 114514
batch_size_train = 64
batch_size_test = 1000
LEARNING_RATE = 0.01
Q_TYPE = "int16"
Q_BITS = 8
MODEL_CACHE = True

device = torch.device("cpu")
epochs = 1

def get_optimizer(model):
  return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)