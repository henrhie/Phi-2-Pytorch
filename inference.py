import torch
from typing import Optional


def multinomial_sample(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits, temperature, top_k)
    idx_next = multinomial_sample(probs)
    return idx_next


def generate(prompt, model, temp=0.0):
    logits, cache = model(prompt[None])
    if temp > 0.0:
        next_token = sample(logits[:, -1, :], temperature=temp)
    else:
        next_token = torch.argmax(logits[:, -1, :], dim=-1)

    yield next_token

    while True:
        logits, cache = model(next_token, cache)
        if temp > 0.0:
            next_token = sample(logits[:, -1, :], temperature=temp)
        else:
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
        yield next_token
