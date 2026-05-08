# Mini GPT — From Scratch in PyTorch

## Overview

This project implements a **decoder-only GPT-style language model from scratch** using Python and PyTorch.

The goal is to deeply understand how Large Language Models (LLMs) work internally by rebuilding the core components manually, including:
- tokenization
- embeddings
- self-attention
- transformer blocks
- training loop
- autoregressive text generation

Unlike high-level libraries, this project focuses on **learning by implementation**, making every part of the model transparent and controllable.

---

## Purpose

This project is designed to:

- Build a strong foundation in transformer architectures
- Understand how GPT models perform next-token prediction
- Gain hands-on experience with:
  - model training
  - loss optimization
  - sequence modeling
- Prepare for real-world LLM workflows (fine-tuning, RAG, deployment)

---

## Tech Stack

- Language: Python 3  
- Framework: PyTorch  
- Libraries:
  - torch  
  - numpy (optional)  

---

## Project Structure

- `src/config.py`: hyperparameters such as `block_size`, `n_embd`, `n_head`, `n_layer`, and `dropout`
- `src/data.py`: character vocabulary, encoding/decoding, train/validation split, and batch sampling
- `src/model.py`: bigram baseline, single-head attention model, and full transformer language model
- `src/train.py`: training loop for the transformer model
- `src/generate.py`: placeholder for loading a trained model and generating from a prompt
- `data/input.txt`: training text

---

## Transformer Core

The current full model is `TransformerLanguageModel` in `src/model.py`.

Input token ids have shape `(B, T)`, where:

- `B` is batch size
- `T` is context length
- `T` must be no larger than `block_size`

The forward pass is:

1. Token embeddings turn ids into vectors: `(B, T) -> (B, T, n_embd)`
2. Positional embeddings give each position an identity, then get added to token embeddings
3. Transformer blocks repeatedly update the sequence representation
4. Final layer norm stabilizes the last representation
5. The language-model head returns next-token logits: `(B, T, vocab_size)`

The transformer pieces are:

- `Head`: one causal self-attention head. Each token can read only previous tokens and itself.
- `MultiHeadAttention`: runs several `Head`s in parallel, concatenates their outputs, then projects back to `n_embd`.
- `FeedForward`: a per-token MLP. Attention mixes information across time; this layer processes each token's updated vector.
- `Block`: one transformer block using layer norm, residual connections, multi-head attention, and feedforward computation.
- `TransformerLanguageModel`: embeddings, a stack of `Block`s, final normalization, output head, loss, and generation.
