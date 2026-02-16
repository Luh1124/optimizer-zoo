"""Small GPT-2 for language modeling on WikiText-2.

A minimal GPT-2 implementation (~10M parameters):
- Vocabulary: 50257 (GPT-2 tokenizer)
- Embedding dim: 384
- Depth: 6 transformer blocks
- Heads: 6
- Context length: 256

Training time: ~20 min on a single GPU for a few epochs.

This is the most realistic benchmark for Muon/SOAP, since these
optimizers are designed for large-scale language model training.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal (masked) multi-head self-attention for autoregressive LM."""

    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Causal mask (upper triangular = -inf)
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj_drop(self.proj(x))


class GPT2Block(nn.Module):
    """GPT-2 transformer block: LN -> CausalAttn -> LN -> MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Small(nn.Module):
    """Small GPT-2 for language modeling (~10M parameters).

    Architecture:
        - vocab_size=50257, embed_dim=384, depth=6, heads=6
        - Learned positional embeddings
        - Weight tying (embedding = lm_head)

    Args:
        vocab_size: Vocabulary size (default: 50257 for GPT-2 tokenizer).
        max_seq_len: Maximum sequence length (default: 256).
        embed_dim: Embedding dimension (default: 384).
        depth: Number of transformer blocks (default: 6).
        num_heads: Number of attention heads (default: 6).
        mlp_ratio: MLP hidden dim ratio (default: 4.0).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 256,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.tok_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            GPT2Block(embed_dim, num_heads, max_seq_len, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)

        # LM head with weight tying
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight

        # Initialize
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for language modeling.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_embed(input_ids) + self.pos_embed(positions))

        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def compute_loss(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for next-token prediction.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).

        Returns:
            Scalar loss.
        """
        logits = self.forward(input_ids)
        # Shift: predict token t+1 from position t
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
