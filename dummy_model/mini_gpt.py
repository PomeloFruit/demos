"""
This is a dummy version of a GPT model implementation in PyTorch. The template
code is from "mini-GPT" by Andrej Karpathy: https://www.youtube.com/watch?v=kCc8FmEb1nY

The goal is to build upon the initial version with more features from modern
transsformer architectures.
"""

import math
import pathlib
import click
import urllib.request
from dataclasses import dataclass

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    # Model architecture
    vocab_size: int = 256  # Byte-level tokenizer
    block_size: int = 256  # Maximum context length (sequence length)
    n_layer: int = 6  # Number of transformer blocks
    n_head: int = 6  # Number of attention heads
    n_embd: int = 384  # Embedding dimensionality
    dropout: float = 0.2  # Dropout rate

    # Training
    batch_size: int = 64  # Sequences per batch
    learning_rate: float = 3e-4  # AdamW learning rate
    max_iters: int = 5000  # Total training steps
    eval_interval: int = 500  # How often to evaluate
    eval_iters: int = 200  # Batches to average over when evaluating

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------------------


class TextDataset:
    """
    Simple byte-level dataset. No external tokenizer needed.

    TODO (Phase 2+): Replace with a BPE tokenizer (tiktoken or sentencepiece)
    for much better token efficiency. Byte-level is fine for learning.
    """

    def __init__(self, text: str, config: GPTConfig):
        self.config = config
        # Byte-level encoding: every character is its own token (0–255)
        self.data = torch.tensor([b for b in text.encode("utf-8")], dtype=torch.long)
        # Train/val split (90/10)
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]
        print(
            f"Dataset: {len(self.data):,} bytes | "
            f"train: {len(self.train_data):,} | val: {len(self.val_data):,}"
        )

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(
            len(data) - self.config.block_size, (self.config.batch_size,)
        )
        x = torch.stack([data[i : i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.config.block_size + 1] for i in ix])
        return x.to(self.config.device), y.to(self.config.device)


# ---------------------------------------------------------------------------
# 3. MODEL COMPONENTS — Each one is a future upgrade target
# ---------------------------------------------------------------------------


class SelfAttention(nn.Module):
    """
    Standard multi-head causal self-attention.

    TODO upgrades:
      - RoPE (Rotary Position Embeddings) instead of learned pos embeddings
      - Grouped Query Attention (GQA) for memory efficiency
      - Flash Attention for speed (torch.nn.functional.scaled_dot_product_attention)
      - KV-cache for fast autoregressive generation
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, Query, Value projections — all heads in one matrix multiply
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Causal mask — prevents attending to future tokens
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Compute Q, K, V for all heads in parallel
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))


class FeedForward(nn.Module):
    """
    Standard MLP: project up 4x, GELU, project back down.

    TODO upgrade: Replace with SwiGLU (used in Llama, Mistral, DeepSeek):
      - Uses gated linear unit with SiLU activation
      - Slightly different expansion ratio (8/3 * n_embd instead of 4x)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=False),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block: attention → add & norm → FFN → add & norm.

    Uses pre-norm (LayerNorm before attention/FFN) which is the modern default.

    TODO upgrade: Replace nn.LayerNorm with RMSNorm for slight speedup.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))  # Residual connection
        x = x + self.ffn(self.ln2(x))  # Residual connection
        return x


# ---------------------------------------------------------------------------
# 4. THE FULL MODEL
# ---------------------------------------------------------------------------


class MiniGPT(nn.Module):
    """
    A small GPT-style language model.

    Architecture: token embeddings + position embeddings → N transformer blocks
    → final layer norm → linear head to predict next token.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(
            config.block_size, config.n_embd
        )  # TODO: Replace with RoPE
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output head.
        # This is a widely-used trick that reduces parameters and improves quality.
        self.token_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    def _init_weights(self, module):
        """Xavier-style init — standard for transformers."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        assert (
            T <= self.config.block_size
        ), f"Sequence length {T} exceeds block_size {self.config.block_size}"

        # Forward pass
        tok_emb = self.token_emb(idx)  # (B, T, n_embd)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Loss (only during training)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with temperature and top-k sampling."""
        for _ in range(max_new_tokens):
            # Crop to block_size if needed
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# ---------------------------------------------------------------------------
# 5. TRAINING LOOP
# ---------------------------------------------------------------------------


@torch.no_grad()
def estimate_loss(model: MiniGPT, dataset: TextDataset, config: GPTConfig) -> dict:
    """Average loss over several batches for more stable evaluation."""
    model.eval()
    losses = {}
    for split in ["train", "val"]:
        batch_losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            x, y = dataset.get_batch(split)
            _, loss = model(x, y)
            batch_losses[k] = loss.item()
        losses[split] = batch_losses.mean().item()
    model.train()
    return losses


def train(config: GPTConfig, dataset: TextDataset) -> MiniGPT:
    model = MiniGPT(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"\nTraining on {config.device} for {config.max_iters} steps...")
    print("-" * 60)

    for step in range(config.max_iters):
        # Periodic evaluation
        if step % config.eval_interval == 0 or step == config.max_iters - 1:
            losses = estimate_loss(model, dataset, config)
            print(
                f"step {step:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f}"
            )

        # Training step
        x, y = dataset.get_batch("train")
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return model


# ---------------------------------------------------------------------------
# 6. MAIN — Downloads data, trains, and generates
# ---------------------------------------------------------------------------


def get_shakespeare() -> str:
    path = "shakespeare.txt"
    if not pathlib.Path(path).exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@click.group()
@click.option(
    "--checkpoint", type=click.Path(), default=None, help="Model checkpoint path"
)
@click.pass_context
def mini_gpt(ctx, checkpoint):
    """Dummy version of GPT model"""
    ctx.ensure_object(dict)
    ctx.obj = {"checkpoint": checkpoint}


@mini_gpt.command()
@click.pass_context
@click.option(
    "--prompt",
    type=str,
    default=None,
    help="Generate from this prompt (requires trained model)",
)
@click.option("--max-tokens", type=int, default=100, help="Tokens to generate")
def inference(ctx, prompt: str, max_tokens: int):
    checkpoint = ctx.obj["checkpoint"]

    config = GPTConfig()
    print(
        f"Mini GPT | {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim"
    )
    print(
        f"Context Window: {config.block_size} tokens | Vocab: {config.vocab_size} byte-level"
    )

    model = MiniGPT(config).to(config.device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=config.device, weights_only=True)
    )
    model.eval()

    prompt_bytes = list(prompt.encode("utf_8"))
    idx = torch.tensor([prompt_bytes], dtype=torch.long, device=config.device)
    output = model.generate(idx, max_new_tokens=max_tokens)
    generated = bytes(output[0].tolist()).decode("utf-8", errors="replace")
    print(f"Generated Text:\n{generated}")
    return


@mini_gpt.command()
@click.pass_context
def train_model(ctx, checkpoint):
    checkpoint = ctx.obj["checkpoint"]

    config = GPTConfig()
    print(
        f"Mini GPT | {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim"
    )
    print(
        f"Context Window: {config.block_size} tokens | Vocab: {config.vocab_size} byte-level"
    )

    text = get_shakespeare()
    dataset = TextDataset(text, config)
    model = train(config, dataset)

    ckpt_path = checkpoint if checkpoint is not None else ""
    torch.save(model.state_dict(), checkpoint)
    print(f"Saved model to {ckpt_path}")
    return


if __name__ == "__main__":
    mini_gpt()
