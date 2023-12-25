import torch
import torch.nn as nn
from pathlib import Path
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoTokenizer
import argparse

from inference import generate


@dataclass
class ModelArgs:
    max_sequence_length: int = 2048
    num_vocab: int = 51200
    model_dim: int = 2560
    num_heads: int = 32
    num_layers: int = 32
    rotary_dim: int = 32


class GELU_Approx(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.60033 * x * (1 + 0.0433603 * x.pow(2)))


class Embedding(nn.Module):
    def __init__(self, n_vocab: int, dims: int, dropout=0.0, device="mps"):
        super().__init__()
        self.wte = nn.Embedding(n_vocab, dims, device=device)
        self.drop = nn.Dropout(dropout, inplace=False)

    def __call__(self, x: torch.Tensor):
        return self.drop(self.wte(x))


class RotaryEncoder(nn.Module):
    def __init__(self, dim: int, base: int = 10000, device: str = "mps"):
        super().__init__()
        self.dim = dim
        self.base = base
        self.device = device

    @staticmethod
    def compute_frequency_theta(x: torch.Tensor, dim: int, offset: int = 0, base: int = 10000,
                                device="mps", dtype=torch.float32):
        # x.shape = B * NH, L, D
        D = dim // 2
        N = x.shape[1] + offset

        dim_pos = torch.arange(0, D, device=device, dtype=dtype)
        pos = torch.arange(offset, N, device=device, dtype=dtype)

        dim_freq = torch.exp(-dim_pos * (math.log(base) / D))
        m_theta = pos.reshape(-1, 1) * dim_freq.reshape(1, -1)
        return torch.cos(m_theta), torch.sin(m_theta)

    def __call__(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, NH, L, D = x.shape
        x = x.reshape(B * NH, L, -1)

        cos_m_theta, sin_m_theta = RotaryEncoder.compute_frequency_theta(x, self.dim, offset,
                                                                         self.base, self.device)
        x1 = x[..., : self.dim // 2]
        x2 = x[..., self.dim // 2: self.dim]
        rx1 = x1 * cos_m_theta - x2 * sin_m_theta
        rx2 = x1 * sin_m_theta + x2 * cos_m_theta
        if self.dim < x.shape[-1]:
            rx = torch.concatenate([rx1, rx2, x[..., self.dim:]], dim=-1)
        else:
            rx = torch.concatenate([rx1, rx2], dim=-1)
        return rx.reshape(B, NH, L, -1)


class Attention(nn.Module):
    def __init__(self, dims: int, rotary_dim: int, n_heads: int, device="mps"):
        super().__init__()

        self.Wqkv = nn.Linear(dims, dims * 3, device=device)
        self.out_proj = nn.Linear(dims, dims, device=device)
        self.rope = RotaryEncoder(dim=rotary_dim, device=device)
        self.n_heads = n_heads
        self.dims = dims

    def forward(self, x: torch.Tensor, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        queries, keys, values = self.Wqkv(x).split(self.dims, dim=-1)

        B, L, _ = queries.shape

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(2, 1)
        keys = keys.reshape(B, L, self.n_heads, -1).transpose(2, 1)
        values = values.reshape(B, L, self.n_heads, -1).transpose(2, 1)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])

            values = torch.concat((value_cache, values), dim=2)
            keys = torch.concat((key_cache, keys), dim=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        queries = queries.type(torch.float32)
        keys = keys.type(torch.float32)

        scale = math.sqrt(1 / queries.shape[-1])
        scores = (scale * queries) @ keys.transpose(3, 2)

        if mask is not None:
            scores += mask

        scores = torch.softmax(scores, dim=-1).type(values.dtype)

        out = scores @ values
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.out_proj(out), (keys, values)


class MLP(nn.Module):
    def __init__(self, dims: int, device="mps"):
        super().__init__()
        self.fc1 = nn.Linear(dims, dims * 4, device=device)
        self.fc2 = nn.Linear(dims * 4, dims, device=device)
        self.activation = GELU_Approx()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class ParallelBlock(nn.Module):
    def __init__(self, args: ModelArgs, device="mps"):
        super().__init__()
        self.args = args

        dims = args.model_dim
        self.ln = nn.LayerNorm(dims, device=device)

        self.mixer = Attention(dims, args.rotary_dim, args.num_heads, device=device)
        self.mlp = MLP(dims, device=device)

    def forward(self, x, mask, cache):
        h = self.ln(x)
        attn_h, cache = self.mixer(h, cache, mask)
        ff_hat = self.mlp(h)
        return attn_h + ff_hat + x, cache


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, device="mps"):
        super().__init__()
        self.embd = Embedding(args.num_vocab, args.model_dim, device=device)
        self.h = nn.ModuleList([ParallelBlock(args, device) for i in range(args.num_layers)])

    def forward(self, x, mask, cache):
        x = self.embd(x)
        if cache is None:
            cache = [None] * len(self.h)

        for e, layer in enumerate(self.h):
            x, cache[e] = layer(x, mask, cache[e])
        return x, cache


class OutputMLP(nn.Module):
    def __init__(self, args: ModelArgs, device="mps"):
        super().__init__()
        self.ln = nn.LayerNorm(args.model_dim, device=device)
        self.linear = nn.Linear(args.model_dim, args.num_vocab, device=device)

    def forward(self, inputs):
        return self.linear(self.ln(inputs))


class Phi2CausalModel(nn.Module):
    def __init__(self, args: ModelArgs, device="mps"):
        super().__init__()
        self.args = args

        self.transformer = Transformer(args, device=device)
        self.lm_head = OutputMLP(args, device=device)
        self.device = device

    @staticmethod
    def create_causal_mask(seq_length: int, device="mps"):
        pos = torch.arange(0, seq_length, device=device)
        mask = pos[:, None] * pos[None]
        return mask * -1e-9

    def forward(self, x: torch.Tensor, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        mask = None
        if x.shape[1] > 1:
            mask = Phi2CausalModel.create_causal_mask(x.shape[1], device=self.device)
            mask = mask.type(x.dtype)

        y, cache = self.transformer(x, mask, cache)
        return self.lm_head(y), cache


def load_weights_and_tokenizer(path: str):
    model_ = Phi2CausalModel(ModelArgs())
    path = Path(path) / 'phi2-weights.pt'
    if not path.exists():
        raise Exception('model weights does not exist in directory ' + str(path))
    print('[INFO] Updating model with weights')
    model_.load_state_dict(torch.load(str(path)))
    tokenizer_ = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    return model_, tokenizer_


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phi-2 Language Model")
    parser.add_argument("--model-path",
                        type=str,
                        default="weights",
                        help="The path to load model weights from")
    parser.add_argument(
        "--prompt",
        "-p",
        help="Prompt input for model to start generation",
        default="What are the seven wonders of the world?",
        type=str,
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=100,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temp",
        "-t",
        help="The sampling temperature.",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Device to run inference on.",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--seed",
        "-s",
        help="The sampling temperature.",
        type=int,
        default=0,
    )

    args_ = parser.parse_args()
    torch.manual_seed(args_.seed)
    model, tokenizer = load_weights_and_tokenizer(args_.model_path)

    prompt = tokenizer(
        args_.prompt,
        return_attention_mask=False,
    )["input_ids"]
    prompt = torch.as_tensor(prompt, dtype=torch.int32, device=args_.device)

    model = model.to(args_.device)

    print("[INFO] Starting Generation", flush=True)
    print(args_.prompt, end="", flush=True)

    tokens = []
    for token, _ in zip(generate(prompt, model, temp=args_.temp), range(args_.max_tokens)):
        tokens.append(token)

        if (len(tokens) % 10) == 0:
            eos_index = next(
                (i for i, t in enumerate(tokens) if t.item() == tokenizer.eos_token_id),
                None,
            )

            if eos_index is not None:
                tokens = tokens[:eos_index]

            s = tokenizer.decode([t.item() for t in tokens])
            print(s, end="", flush=True)
            tokens = []
            if eos_index is not None:
                break

    s = tokenizer.decode([t.item() for t in tokens])
    print(s, flush=True)

