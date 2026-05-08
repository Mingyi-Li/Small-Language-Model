# Token embedding
# Positional embedding
# Attention head
# Multi-head attention
# Feedforward block
# Transformer block
# Full model

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import block_size, n_embd, n_head, n_layer, dropout

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx: (B, T)  = (batch_size, block_size)
        B, T = idx.shape

        # (B, T, n_embd), turn the 2D "idx" matrix to 3D, real amount of learning features 
        x = self.token_embedding_table(idx)

        # (B, T, vocab_size), output of probs of next chars
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits = logits.view(B * T, -1)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
class Head(nn.Module):
    """
    One head of masked self-attention.
    """

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (B, T, C), C is n_embd
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores.
        # k^T: (B, head_size, T)
        # q @ k^T gives shape (B, T, T)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        # Causal mask: tokens cannot see future tokens
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Convert scores into probabilities
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)  # (B, T, head_size)

        # Weighted sum of values
        out = wei @ v      # (B, T, head_size)

        return out

class MultiHeadAttention(nn.Module):
    """
    Multiple causal attention heads running in parallel.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Each head returns (B, T, head_size); concat gives (B, T, n_embd).
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """
    Per-token MLP applied after attention.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Transformer block: communication followed by per-token computation.
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual connections let each block add useful updates to x.
        # new x = old x + attention update
        x = x + self.sa(self.ln1(x))
        # new x = old x + feedforward update
        x = x + self.ffwd(self.ln2(x))
        return x

class SelfAttentionLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: (B, T)
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)  # (B, T, C)

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos) # (T, C)

        x = token_emb + pos_emb                      # (B, T, C)
        x = self.sa_head(x)                          # (B, T, C)

        logits = self.lm_head(x)                     # (B, T, vocab_size)

        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            # Crop context so it does not exceed block_size
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            # Use only the final time step
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

class TransformerLanguageModel(nn.Module):
    """
    Decoder-only GPT-style language model.
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx shape: (B, T)
        B, T = idx.shape
        if T > block_size:
            raise ValueError(f"Cannot forward sequence of length {T}; block_size is {block_size}")

        token_emb = self.token_embedding_table(idx)  # (B, T, C)

        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos) # (T, C)

        x = token_emb + pos_emb                      # (B, T, C)
        x = self.blocks(x)                           # (B, T, C)
        x = self.ln_f(x)                             # (B, T, C)

        logits = self.lm_head(x)                     # (B, T, vocab_size)

        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):

            # Crop context so it does not exceed block_size
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            # Use only the final time step
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
