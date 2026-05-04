# Store hyperparameters
# Batch size, block size, learning rate, embedding size, number of heads, number of layers, dropout, training steps

import torch

batch_size = 32
block_size = 64
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 100

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"