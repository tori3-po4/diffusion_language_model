"""DIT Transformer for MDLM with adaLN-Zero, RoPE, and SUBS parameterization."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → MLP → conditioning vector."""

    def __init__(self, hidden_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.frequency_dim = frequency_dim

    def sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) noise level sigma → (B, hidden_dim)."""
        emb = self.sinusoidal_embedding(t)
        return self.mlp(emb)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to query and key tensors.

    q, k: (B, n_heads, S, head_dim)
    cos, sin: (S, head_dim)
    """
    cos = cos[None, None, :, :]  # (1, 1, S, head_dim)
    sin = sin[None, None, :, :]
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


class DDiTBlock(nn.Module):
    """Transformer block with adaLN-Zero modulation.

    adaLN produces 6 parameters: shift/scale/gate for both attention and FFN.
    Gates are initialized to zero for training stability.
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        # adaLN modulation: 6 parameters (shift1, scale1, gate1, shift2, scale2, gate2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        # Initialize gate parameters to zero
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        # Self-attention
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, S, D), c: (B, D) conditioning, cos/sin: (S, head_dim) RoPE.
        """
        B, S, D = x.shape

        # adaLN modulation parameters
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN_modulation(c).chunk(6, dim=-1)

        # Attention branch
        h = self._modulate(self.norm1(x), shift1, scale1)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, S, head_dim)
        q, k, v = qkv.unbind(0)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention (bidirectional, no causal mask)
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, D)
        attn_out = self.out_proj(attn_out)
        attn_out = self.attn_dropout(attn_out)

        x = x + gate1.unsqueeze(1) * attn_out

        # FFN branch
        h = self._modulate(self.norm2(x), shift2, scale2)
        x = x + gate2.unsqueeze(1) * self.ffn(h)

        return x


class DDiTFinalLayer(nn.Module):
    """Final layer: adaLN + linear projection to vocab."""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.linear = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class DIT(nn.Module):
    """Diffusion Transformer for MDLM.

    Architecture: token embedding + timestep embedding → N x DDiTBlock → final layer → logits.
    Uses weight tying between input embedding and output projection.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        mask_token_id: int = 50257,
        hidden_dim: int = 384,
        n_layers: int = 6,
        n_heads: int = 6,
        seq_len: int = 256,
        dropout: float = 0.0,
        weight_tying: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.full_vocab_size = vocab_size + 1  # +1 for [MASK]
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(self.full_vocab_size, hidden_dim)
        self.timestep_embedder = TimestepEmbedder(hidden_dim)
        self.rotary_emb = RotaryEmbedding(hidden_dim // n_heads, max_seq_len=seq_len)

        self.blocks = nn.ModuleList([
            DDiTBlock(hidden_dim, n_heads, dropout) for _ in range(n_layers)
        ])

        self.final_layer = DDiTFinalLayer(hidden_dim, self.full_vocab_size)

        if weight_tying:
            self.final_layer.linear.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        # Standard initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        # Re-zero adaLN modulation layers (adaLN-Zero: gates must start at zero
        # so each block acts as identity at initialization for stable training)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[1].weight)
            nn.init.zeros_(block.adaLN_modulation[1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[1].bias)

    def forward(self, indices: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: Token indices (B, S), may contain mask_token_id.
            sigma: Noise level per sample (B,).

        Returns:
            logits: (B, S, full_vocab_size).
        """
        x = self.token_embedding(indices)
        c = self.timestep_embedder(sigma)

        S = indices.shape[1]
        cos, sin = self.rotary_emb(S)

        for block in self.blocks:
            x = block(x, c, cos, sin)

        logits = self.final_layer(x, c)
        return logits


def subs_parameterization(
    logits: torch.Tensor,
    xt: torch.Tensor,
    x0: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    """SUBS parameterization: compute NLL only at [MASK] positions.

    Memory-efficient version: only computes cross-entropy at masked positions,
    avoiding materializing a full (B, S, V) float32 tensor.

    Args:
        logits: Raw model output (B, S, V).
        xt: Noisy input tokens (B, S).
        x0: Clean target tokens (B, S).
        mask_token_id: The [MASK] token ID.

    Returns:
        nll: Per-position negative log-likelihood (B, S). Zero at non-masked positions.
    """
    is_mask = (xt == mask_token_id)  # (B, S)
    B, S = xt.shape

    nll = torch.zeros(B, S, device=xt.device, dtype=torch.float32)

    if is_mask.any():
        # Extract logits/targets only at masked positions (much smaller than full B*S*V)
        masked_logits = logits[is_mask]   # (num_masked, V)
        masked_targets = x0[is_mask]      # (num_masked,)

        # F.cross_entropy fuses softmax+gather internally, avoiding full softmax allocation
        masked_nll = F.cross_entropy(
            masked_logits.float(), masked_targets, reduction="none"
        )
        nll[is_mask] = masked_nll

    return nll
