# Store hyperparameters
# Batch size, block size, learning rate, embedding size, number of heads, number of layers, dropout, training steps

import torch
# # of text chunks selected to train at each time, # of rows in the matrix "idx"
batch_size = 32
# length of the text chunk, # of columns in the matrix "idx"
block_size = 64
max_iters = 3000
eval_interval = 300
learning_rate = 3e-4
eval_iters = 100

# the length of the vector each token is turn into
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2

device = "cuda" if torch.cuda.is_available() else "cpu"