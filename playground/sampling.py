import torch
from torch import Tensor

def batch_greedy_sampling(logits: Tensor):
    return logits.argmax(dim=-1, keepdim=True)

def temperature_sampling(logits: Tensor, T: float):
    logits = logits / T
    prob = logits.softmax(dim=-1)
    return torch.multinomial(prob, num_samples=1)

def top_k_sampling(logits: Tensor, T: float, k: int):
    logits = logits / T
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    prob = torch.softmax(sorted_logits[..., :k], dim=-1)
    choice = torch.multinomial(prob, num_samples=1)
    return torch.gather(sorted_indices, dim=-1, index=choice)

def top_p_sampling(logits: Tensor, T: float, p: float):
    logits = logits / T
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    prob = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(prob, dim=-1)
    remove = cum >= p

    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    sorted_logits[remove] = -float('inf')
    new_prob = torch.softmax(sorted_logits, dim=-1)
    choices = torch.multinomial(new_prob, num_samples=1)
    return torch.gather(sorted_indices, dim=-1, index=choices)

def top_k_top_p_sampling(logits: Tensor, T: float, k: int, p: float):
    logits = logits / T
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    prob = torch.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(prob, dim=-1)

    remove = cum >= p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    remove[..., k:] = True

    prob[remove] = -float('inf')
    new_prob = torch.softmax(prob, dim=-1)
    choices = torch.multinomial(new_prob, num_samples=1)
    return torch.gather(sorted_indices, dim=-1, index=choices)

def main():
    n_tokens = 4
    dim = 8
    logits = torch.randn(n_tokens, dim)
    print(logits)
    print(batch_greedy_sampling(logits))
    print(temperature_sampling(logits, 0.01))
    print(top_k_sampling(logits, 0.01, 1))
    print(top_p_sampling(logits, 0.01, 0.01))
    print(top_k_top_p_sampling(logits, 0.01, 1, 0.01))

if __name__ == "__main__":
    main()