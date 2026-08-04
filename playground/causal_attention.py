import torch
from torch import Tensor
import torch.nn.functional as F
import math

def causal_attention(q: Tensor, k: Tensor, v: Tensor):
    n, d = q.shape
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    multi = (q @ k.T) / math.sqrt(d)
    multi = multi.masked_fill(mask, -float('inf'))
    score = torch.softmax(multi, dim=-1)
    return score @ v

def main():
    n_tokens = 32
    dim = 16
    q, k, v = torch.randn(n_tokens, dim), torch.randn(n_tokens, dim), torch.randn(n_tokens, dim)
    mine = causal_attention(q, k, v)
    tor = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print(torch.allclose(mine, tor, atol=1e-5, rtol=1e-5))

if __name__ == "__main__":
    main()