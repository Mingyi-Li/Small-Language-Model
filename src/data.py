# Reading text , build volcabulary, char <=> int encoding
# Train / validation split
# Sampling batches

import torch
from config import batch_size, block_size

# Read input.txt
with open("../data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Dataset length:", len(text))

# Sort all input chars
chars = sorted(list(set(text)))
vocab_size = len(chars)

# print("Vocab size:", vocab_size)
# print(chars)

# Encoding/Decoding
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

test = "Hello"
encoded = encode(test)
decoded = decode(encoded)

# print("Encoded:", encoded)
# print("Decoded:", decoded)

# Convert to tensor
data = torch.tensor(encode(text), dtype=torch.long)
print("data shape:", data.shape)

# Train / validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# print("Train size:", train_data.shape)
# print("Val size:", val_data.shape)

def get_batch(split):
    data_source = train_data if split == "train" else val_data
    
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    
    return x, y

x, y = get_batch("train")

# print("x shape:", x.shape)
# print("y shape:", y.shape)

# print("x sample:", x[0])
# print("y sample:", y[0])

# # Decode
# print("x decoded:")
# print(decode(x[0].tolist()))

# print("y decoded:")
# print(decode(y[0].tolist()))