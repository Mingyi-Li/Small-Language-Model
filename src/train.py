# Model creation
# Optimizer
# Training loop
# Loss calculation
# Validation loss
# Saving checkpoints

import torch
from model import TransformerLanguageModel
from data import get_batch, vocab_size, decode
from config import device, learning_rate, max_iters

model = TransformerLanguageModel(vocab_size)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)

    # Reset gradients
    optimizer.zero_grad()
    # Backward prop
    loss.backward()
    # Update gradients
    optimizer.step()

    if iter % 100 == 0:
        print(f"step {iter}: loss {loss.item():.4f}")

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=300)

print(decode(generated[0].tolist()))
