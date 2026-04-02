"""
4D Cross-Attention + RoPE for SigLIP (queries) ← DINO (keys/values)

- Inputs:
    SigLIP  : [B, T, Ns, Ds]
    DINO    : [B, T, Nd, Dd]
- Output:
    sig_fused: [B, T, Ns, D_model]   # (optionally also din_fused if bidirectional)

Notes:
- RoPE rotates q/k at shape [B, H, T, N, d] using coords [B, T, N, 1+coord_dim] = (t, y, x) ∈ [0,1].
- Attention is computed **per time step** (T independent SDPA calls via flattening B*T as batch).
- If you want cross-time attention, you can extend to a window/global mode; this version is frame-local and fast/stable.
"""

from typing import Optional, Tuple, Literal, List, Sequence
import math
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------
# helpers
# ---------------------------

def infer_hw_from_n(n: int) -> Tuple[int, int]:
    r = int(math.sqrt(n))
    return (r, r) if r * r == n else (1, n)

def make_coords_3d_4d(
    B: int, T: int, N: int, device: torch.device, hw: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    Build coords [B, T, N, 3] with (t, y, x) in [0,1].
    """
    if hw is None:
        H, W = infer_hw_from_n(N)
    else:
        H, W = hw
        assert H * W == N, f"H*W={H*W} must equal N={N}"

    t = torch.linspace(0.0, 1.0, T, device=device)                     # [T]
    y = torch.linspace(0.0, 1.0, H, device=device)                     # [H]
    x = torch.linspace(0.0, 1.0, W, device=device)                     # [W]
    yy, xx = torch.meshgrid(y, x, indexing="ij")                       # [H, W]
    # [T, H, W, 3]
    coords = torch.stack([
        t.view(T, 1, 1).expand(T, H, W),
        yy.expand(T, H, W),
        xx.expand(T, H, W)
    ], dim=-1)
    coords = coords.view(T, H * W, 3)                                  # [T, N, 3]
    coords = coords.unsqueeze(0).expand(B, T, N, 3).contiguous()       # [B, T, N, 3]
    return coords


# ---------------------------
# RoPE 4D (B,H,T,N,d) with separate q/k coords
# ---------------------------

def _pos_embed_fourier1d_init(cutoff: float = 256, n: int = 32) -> torch.Tensor:
    # same spirit as your code; max initial frequency ~1, down to ~1/cutoff
    return torch.exp(torch.linspace(0, -math.log(cutoff), n)).unsqueeze(0).unsqueeze(0)

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

class RotaryPositionalEncoding4D(nn.Module):
    """
    4D RoPE: coords_q/k are [B, T, N, D_pos] where D_pos = 1+coord_dim (t,y,x).
    n_pos is a tuple of EVEN integers whose sum equals per-head dim d.
    """
    def __init__(self, cutoffs: Tuple[float, ...], n_pos: Tuple[int, ...]):
        super().__init__()
        assert len(cutoffs) == len(n_pos)
        if not all((n % 2 == 0) and (n > 0) for n in n_pos):
            raise ValueError("each n_pos must be positive and EVEN")
        self.freqs = nn.ParameterList([
            nn.Parameter(_pos_embed_fourier1d_init(cutoff, n // 2))
            for cutoff, n in zip(cutoffs, n_pos)
        ])

    def _co_si(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # coords: [B, T, N, Dp]
        B, T, N, Dp = coords.shape
        assert Dp == len(self.freqs)
        parts_co, parts_si = [], []
        for axis, freq in enumerate(self.freqs):
            # coords[..., axis:axis+1]: [B,T,N,1]; freq: [1,1,n//2]
            arg = 0.5 * math.pi * coords[..., axis:axis+1] * freq
            # normalization ~ 1/sqrt(#freq per axis) like your code
            scale = 1.0 / math.sqrt(len(freq))
            parts_co.append(torch.cos(arg) * scale)
            parts_si.append(torch.sin(arg) * scale)
        co = torch.cat(parts_co, dim=-1)  # [B,T,N, d/2]
        si = torch.cat(parts_si, dim=-1)  # [B,T,N, d/2]
        return co, si

    def forward(
        self,
        q: torch.Tensor,             # [B, H, T, Nq, d]
        k: torch.Tensor,             # [B, H, T, Nk, d]
        coords_q: torch.Tensor,      # [B, T, Nq, D_pos]
        coords_k: Optional[torch.Tensor] = None,  # [B, T, Nk, D_pos]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if coords_k is None: coords_k = coords_q
        co_q, si_q = self._co_si(coords_q)  # [B,T,Nq,d/2]
        co_k, si_k = self._co_si(coords_k)  # [B,T,Nk,d/2]

        # expand to heads and repeat_interleave along the "pair" axis
        co_q = co_q.unsqueeze(1).repeat_interleave(2, dim=-1)  # [B,H,T,Nq,d]
        si_q = si_q.unsqueeze(1).repeat_interleave(2, dim=-1)
        co_k = co_k.unsqueeze(1).repeat_interleave(2, dim=-1)  # [B,H,T,Nk,d]
        si_k = si_k.unsqueeze(1).repeat_interleave(2, dim=-1)

        q2 = q * co_q + _rotate_half(q) * si_q
        k2 = k * co_k + _rotate_half(k) * si_k
        return q2, k2


# ---------------------------
# Cross-Attention 4D (frame-local) + RoPE
# ---------------------------

class RelativePositionalCrossAttention4D(nn.Module):
    """
    Frame-local cross-attention:
      - Q: SigLIP  [B,T,Nq,D]
      - K,V: DINO  [B,T,Nk,D]
      - RoPE on [B,H,T,N, d] with coords [B,T,N, 1+coord_dim]
      - Optional spatial cutoff / distance prior (per frame)

    Shapes preserved: output -> [B, T, Nq, D]
    """
    def __init__(
        self,
        coord_dim: int,                 # usually 2 (y,x)
        embed_dim: int,
        n_head: int,
        cutoff_spatial: Optional[float] = None,   # in [0,1] units (since coords normalized)
        dropout: float = 0.0,
        use_rope: bool = True,
        attn_dist_mode: Literal["none", "v0", "v1"] = "none",
        cutoff_temporal: float = 256.0,           # RoPE frequency cutoff for t
        cutoff_spatial_rope: float = 256.0,       # RoPE frequency cutoff for y/x
    ):
        super().__init__()
        if embed_dim % (2 * n_head) != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by 2*n_head {2*n_head}")

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.dropout = dropout
        self.cutoff_spatial = cutoff_spatial
        self.attn_dist_mode = attn_dist_mode

        self.use_rope = use_rope
        if use_rope:
            d = embed_dim // n_head                      # per-head dim
            n_split = 2 * (d // (2 * (coord_dim + 1)))  # even portion for each spatial axis
            n_t = d - coord_dim * n_split               # remaining for time, stays even
            n_pos = (n_t,) + (n_split,) * coord_dim
            cutoffs = (cutoff_temporal,) + (cutoff_spatial_rope,) * coord_dim
            self.rope = RotaryPositionalEncoding4D(cutoffs=cutoffs, n_pos=n_pos)

        self.alpha = nn.Parameter(torch.zeros(n_head))   # gain for distance prior

    def forward(
        self,
        q_tokens: torch.Tensor,         # [B, T, Nq, D]
        kv_tokens: torch.Tensor,        # [B, T, Nk, D]
        coords_q: torch.Tensor,         # [B, T, Nq, 1+coord_dim]
        coords_k: torch.Tensor,         # [B, T, Nk, 1+coord_dim]
        q_pad_mask: Optional[torch.Tensor] = None,    # [B, T, Nq] bool
        kv_pad_mask: Optional[torch.Tensor] = None,   # [B, T, Nk] bool
    ) -> torch.Tensor:
        B, T, Nq, D = q_tokens.shape
        _, _, Nk, D2 = kv_tokens.shape
        assert D == D2, "q and kv channel dims must match"

        H, d = self.n_head, D // self.n_head

        # Project
        q = self.q_proj(q_tokens).view(B, T, Nq, H, d).permute(0, 3, 1, 2, 4)  # [B,H,T,Nq,d]
        k = self.k_proj(kv_tokens).view(B, T, Nk, H, d).permute(0, 3, 1, 2, 4) # [B,H,T,Nk,d]
        v = self.v_proj(kv_tokens).view(B, T, Nk, H, d).permute(0, 3, 1, 2, 4) # [B,H,T,Nk,d]

        # RoPE on 4D
        if self.use_rope:
            q, k = self.rope(q, k, coords_q, coords_k)  # still [B,H,T,N, d]

        # Build additive mask/prior per frame
        dtype = q.dtype
        attn_add = None
        if (self.cutoff_spatial is not None) or (self.attn_dist_mode != "none") or (q_pad_mask is not None) or (kv_pad_mask is not None):
            attn_add = torch.zeros((B, H, T, Nq, Nk), device=q.device, dtype=dtype)

            if self.cutoff_spatial is not None:
                dist_sp = torch.cdist(coords_q[..., -2:], coords_k[..., -2:])      # [B,T,Nq,Nk]
                attn_add = attn_add.masked_fill(
                    dist_sp.unsqueeze(1) > self.cutoff_spatial,
                    torch.finfo(dtype).min / 2,
                )

            if self.attn_dist_mode != "none":
                if self.attn_dist_mode == "v0":
                    dist = torch.cdist(coords_q, coords_k)                         # [B,T,Nq,Nk] over (t,y,x)
                    prior = torch.exp(-0.1 * dist).unsqueeze(1)                    # [B,1,T,Nq,Nk]
                elif self.attn_dist_mode == "v1":
                    dist = torch.cdist(coords_q[..., -2:], coords_k[..., -2:])     # spatial only
                    prior = torch.exp(-5.0 * dist / (self.cutoff_spatial or 1.0)).unsqueeze(1)
                else:
                    raise ValueError(f"Unknown attn_dist_mode {self.attn_dist_mode}")
                attn_add = attn_add + self.alpha.view(1, H, 1, 1, 1) * prior

            if (q_pad_mask is not None) or (kv_pad_mask is not None):
                if q_pad_mask is None:
                    q_pad_mask = torch.zeros((B, T, Nq), dtype=torch.bool, device=q.device)
                if kv_pad_mask is None:
                    kv_pad_mask = torch.zeros((B, T, Nk), dtype=torch.bool, device=q.device)
                ignore = torch.logical_or(
                    q_pad_mask.unsqueeze(-1),
                    kv_pad_mask.unsqueeze(-2)
                ).unsqueeze(1)  # [B,1,T,Nq,Nk]
                attn_add = attn_add.masked_fill(ignore, torch.finfo(dtype).min / 2)

        # Frame-local SDPA: flatten B and T as batch
        q_bt = q.permute(0, 2, 1, 3, 4).reshape(B * T, H, Nq, d)   # [B*T,H,Nq,d]
        k_bt = k.permute(0, 2, 1, 3, 4).reshape(B * T, H, Nk, d)   # [B*T,H,Nk,d]
        v_bt = v.permute(0, 2, 1, 3, 4).reshape(B * T, H, Nk, d)   # [B*T,H,Nk,d]

        if attn_add is not None:
            attn_add_bt = attn_add.permute(0, 2, 1, 3, 4).reshape(B * T, H, Nq, Nk)  # [B*T,H,Nq,Nk]
        else:
            attn_add_bt = None

        y_bt = F.scaled_dot_product_attention(
            q_bt, k_bt, v_bt,
            attn_mask=attn_add_bt,
            dropout_p=self.dropout if self.training else 0.0
        )  # [B*T,H,Nq,d]

        # restore to [B, T, Nq, D]
        y = y_bt.transpose(1, 2).contiguous().view(B * T, Nq, D)
        y = y.view(B, T, Nq, D)
        return self.out_proj(y)


# ---------------------------
# End-to-end fusion module (4D in/out)
# ---------------------------

class SigLIP_DINO_CrossRoPE4D(nn.Module):
    def __init__(
        self,
        d_siglip: int = 1152,
        d_dino: int = 1024,
        d_model: int = 1024,
        n_head: int = 8,
        cutoff_spatial_mask: Optional[float] = None,  # e.g., 0.25 (normalized)
        dropout: float = 0.0,
        attn_dist_mode: Literal["none", "v0", "v1"] = "none",
        bidirectional: bool = False,
        rope_cutoff_t: float = 256.0,
        rope_cutoff_xy: float = 256.0,
    ):
        super().__init__()
        assert d_model % n_head == 0

        self.proj_sig = nn.Linear(d_siglip, d_model, bias=True)
        self.proj_din = nn.Linear(d_dino, d_model, bias=True)
        self.ln_sig = nn.LayerNorm(d_model)
        self.ln_din = nn.LayerNorm(d_model)

        self.cross_sig_from_dino = RelativePositionalCrossAttention4D(
            coord_dim=2, embed_dim=d_model, n_head=n_head,
            cutoff_spatial=cutoff_spatial_mask, dropout=dropout,
            use_rope=True, attn_dist_mode=attn_dist_mode,
            cutoff_temporal=rope_cutoff_t, cutoff_spatial_rope=rope_cutoff_xy,
        )
        self.gate_sig = nn.Parameter(torch.zeros(1))

        self.bidirectional = bidirectional
        if bidirectional:
            self.cross_dino_from_sig = RelativePositionalCrossAttention4D(
                coord_dim=2, embed_dim=d_model, n_head=n_head,
                cutoff_spatial=cutoff_spatial_mask, dropout=dropout,
                use_rope=True, attn_dist_mode=attn_dist_mode,
                cutoff_temporal=rope_cutoff_t, cutoff_spatial_rope=rope_cutoff_xy,
            )
            self.gate_din = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        feat_siglip: torch.Tensor,                              # [B,T,Ns,Ds]
        feat_dino: torch.Tensor,                                # [B,T,Nd,Dd]
        coords_siglip: Optional[torch.Tensor] = None,           # [B,T,Ns,3]
        coords_dino: Optional[torch.Tensor] = None,             # [B,T,Nd,3]
        hw_siglip: Optional[Tuple[int, int]] = None,
        hw_dino: Optional[Tuple[int, int]] = None,
        use_time: bool = True,
    ):
        B, T, Ns, Ds = feat_siglip.shape
        B2, T2, Nd, Dd = feat_dino.shape
        assert (B == B2) and (T == T2), "SigLIP and DINO must match in B and T"

        device = feat_siglip.device

        sig = self.ln_sig(self.proj_sig(feat_siglip))  # [B,T,Ns,Dm]
        din = self.ln_din(self.proj_din(feat_dino))    # [B,T,Nd,Dm]

        if coords_siglip is None:
            coords_siglip = make_coords_3d_4d(B, T, Ns, device, hw_siglip)  # [B,T,Ns,3]
        if coords_dino is None:
            coords_dino = make_coords_3d_4d(B, T, Nd, device, hw_dino)      # [B,T,Nd,3]

        if not use_time:
            coords_siglip = coords_siglip.clone(); coords_siglip[..., 0] = 0.0
            coords_dino   = coords_dino.clone();   coords_dino[..., 0]   = 0.0

        delta_sig = self.cross_sig_from_dino(
            q_tokens=sig, kv_tokens=din,
            coords_q=coords_siglip, coords_k=coords_dino,
            q_pad_mask=None, kv_pad_mask=None,
        )  # [B,T,Ns,Dm]

        sig_fused = sig + torch.tanh(self.gate_sig) * delta_sig

        outputs = {"sig_fused": sig_fused, "coords_sig": coords_siglip}

        if self.bidirectional:
            delta_din = self.cross_dino_from_sig(
                q_tokens=din, kv_tokens=sig,
                coords_q=coords_dino, coords_k=coords_siglip,
                q_pad_mask=None, kv_pad_mask=None,
            )
            din_fused = din + torch.tanh(self.gate_din) * delta_din
            outputs.update({"din_fused": din_fused, "coords_din": coords_dino})

        return outputs




"""
RoPE-only layer pooling (no alpha/beta weights, no MLPs).

For each stream (e.g., SigLIP or DINO):
- Pick a subset of layer tensors by 1-based indices `chosen`.
- Choose one of those as the query "anchor" layer `anchor_idx`.
- Build (t,y,x,l) coordinates where `l` encodes the normalized rank of the layer
  within the chosen subset.
- Run RoPE cross-attention: anchor ← concat(all chosen layers).
- Output shape matches the anchor layer: [T, N, D] (or [B, T, N, D] if inputs had batch).

Usage:
    pool_sig = LayerPoolRoPE(d_model=Ds, n_head=8)
    sig_pooled = pool_sig(image_features, idx1, anchor1, hw=(24,24))

    pool_din = LayerPoolRoPE(d_model=Dd, n_head=8)
    din_pooled = pool_din(image_features2, idx2, anchor2, hw=(14,14))
"""




# ---------------------------
# small helpers
# ---------------------------

def _infer_hw_from_n(n: int) -> Tuple[int, int]:
    r = int(math.sqrt(n))
    return (r, r) if r * r == n else (1, n)

def _normalize_index_position(idx: int, chosen: Sequence[int]) -> float:
    """Map a 1-based layer index to [0,1] using RANK within the chosen subset."""
    pos = chosen.index(idx)  # 0-based rank inside chosen
    L = max(1, len(chosen) - 1)
    return 0.0 if len(chosen) == 1 else (pos / L)

def _ensure_btnd(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Accept [T,N,D] or [B,T,N,D]. Return [B,T,N,D], and a flag if B was added."""
    if x.dim() == 3:
        return x.unsqueeze(0), True
    assert x.dim() == 4, f"expected [T,N,D] or [B,T,N,D], got {list(x.shape)}"
    return x, False


# ---------------------------
# RoPE: ND (here D_pos = 4: t,y,x,l)
# ---------------------------

def _pos_embed_fourier1d_init(cutoff: float = 256, n: int = 32) -> torch.Tensor:
    # frequencies in [~1/cutoff, 1]
    return torch.exp(torch.linspace(0, -math.log(cutoff), n)).unsqueeze(0).unsqueeze(0)

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)

class RotaryPositionalEncodingND(nn.Module):
    """
    Generic ND RoPE (number of axes = len(cutoffs) = len(n_pos)).
    Each n_pos[i] must be EVEN; sum(n_pos) == per-head dim.
    """
    def __init__(self, cutoffs: Tuple[float, ...], n_pos: Tuple[int, ...]):
        super().__init__()
        assert len(cutoffs) == len(n_pos)
        if not all((n % 2 == 0) and (n > 0) for n in n_pos):
            raise ValueError("every n_pos must be positive and EVEN")
        self.freqs = nn.ParameterList([
            nn.Parameter(_pos_embed_fourier1d_init(c, n // 2))
            for c, n in zip(cutoffs, n_pos)
        ])

    def _co_si(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # coords: [B, T, N, D_pos]
        B, T, N, Dp = coords.shape
        assert Dp == len(self.freqs)
        parts_co, parts_si = [], []
        for axis, freq in enumerate(self.freqs):
            arg = 0.5 * math.pi * coords[..., axis:axis+1] * freq  # [B,T,N,n//2]
            scale = 1.0 / math.sqrt(len(freq))
            parts_co.append(torch.cos(arg) * scale)
            parts_si.append(torch.sin(arg) * scale)
        co = torch.cat(parts_co, dim=-1)  # [B,T,N, d/2]
        si = torch.cat(parts_si, dim=-1)  # [B,T,N, d/2]
        return co, si

    def forward(
        self,
        q: torch.Tensor,             # [B, H, T, Nq, d]
        k: torch.Tensor,             # [B, H, T, Nk, d]
        coords_q: torch.Tensor,      # [B, T, Nq, D_pos]
        coords_k: Optional[torch.Tensor] = None,  # [B, T, Nk, D_pos]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if coords_k is None:
            coords_k = coords_q
        co_q, si_q = self._co_si(coords_q)  # [B,T,Nq,d/2]
        co_k, si_k = self._co_si(coords_k)  # [B,T,Nk,d/2]
        co_q = co_q.unsqueeze(1).repeat_interleave(2, dim=-1)  # [B,H,T,Nq,d]
        si_q = si_q.unsqueeze(1).repeat_interleave(2, dim=-1)
        co_k = co_k.unsqueeze(1).repeat_interleave(2, dim=-1)  # [B,H,T,Nk,d]
        si_k = si_k.unsqueeze(1).repeat_interleave(2, dim=-1)
        q2 = q * co_q + _rotate_half(q) * si_q
        k2 = k * co_k + _rotate_half(k) * si_k
        return q2, k2


# ---------------------------
# Cross-attention (frame-local) + RoPE, used internally for pooling
# ---------------------------

def _even_splits(d: int, n_axes: int) -> Tuple[int, ...]:
    """Evenly split per-head dim d across n_axes, each part even; sum(parts) == d."""
    base = (d // (2 * n_axes)) * 2  # even
    parts = [base] * n_axes
    rem = d - sum(parts)
    i = 0
    while rem > 0:
        parts[i] += 2
        rem -= 2
        i = (i + 1) % n_axes
    return tuple(parts)

class CrossAttentionRoPE4D(nn.Module):
    """
    Frame-local RoPE cross-attention without extra priors.

    q_tokens: [B,T,Nq,D]
    kv_tokens:[B,T,Nk,D]
    coords_q: [B,T,Nq, D_pos]   (D_pos=4: t,y,x,l)
    coords_k: [B,T,Nk, D_pos]
    returns : [B,T,Nq,D]
    """
    def __init__(
        self,
        embed_dim: int,
        n_head: int,
        d_pos: int = 4,                          # (t,y,x,l)
        rope_cutoffs: Optional[Tuple[float, ...]] = None,  # per-axis cutoffs
    ):
        super().__init__()
        assert embed_dim % (2 * n_head) == 0
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        d = embed_dim // n_head
        n_pos = _even_splits(d, d_pos)  # even per-axis shares summing to d
        if rope_cutoffs is None:
            rope_cutoffs = (256.0,) * d_pos
        assert len(rope_cutoffs) == d_pos
        self.rope = RotaryPositionalEncodingND(cutoffs=rope_cutoffs, n_pos=n_pos)

    def forward(
        self,
        q_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        coords_q: torch.Tensor,
        coords_k: torch.Tensor,
    ) -> torch.Tensor:
        B, T, Nq, D = q_tokens.shape
        _, _, Nk, D2 = kv_tokens.shape
        assert D == D2

        H, d = self.n_head, D // self.n_head

        q = self.q_proj(q_tokens).view(B, T, Nq, H, d).permute(0, 3, 1, 2, 4)   # [B,H,T,Nq,d]
        k = self.k_proj(kv_tokens).view(B, T, Nk, H, d).permute(0, 3, 1, 2, 4)  # [B,H,T,Nk,d]
        v = self.v_proj(kv_tokens).view(B, T, Nk, H, d).permute(0, 3, 1, 2, 4)  # [B,H,T,Nk,d]

        q, k = self.rope(q, k, coords_q, coords_k)  # RoPE only

        # SDPA per-frame: flatten (B,T) -> BT
        q_bt = q.permute(0, 2, 1, 3, 4).reshape(B * T, H, Nq, d)
        k_bt = k.permute(0, 2, 1, 3, 4).reshape(B * T, H, Nk, d)
        v_bt = v.permute(0, 2, 1, 3, 4).reshape(B * T, H, Nk, d)

        y_bt = F.scaled_dot_product_attention(q_bt, k_bt, v_bt, attn_mask=None, dropout_p=0.0)
        y = y_bt.transpose(1, 2).contiguous().view(B, T, Nq, D)  # [B,T,Nq,D]
        return self.out_proj(y)


# ---------------------------
# Layer pooling via RoPE
# ---------------------------

class LayerPoolRoPE(nn.Module):
    """
    Given a list of layers (chosen indices), produce a SINGLE fused tensor with the SAME shape
    as the chosen 'anchor' layer: [T, N, D] (or [B, T, N, D]).

    Implementation: queries = anchor layer; keys/values = all chosen layers collapsed (L*N).
    Coords use (t,y,x,l) with l ∈ [0,1] the normalized rank within the chosen set.
    """
    def __init__(self, d_model: int, n_head: int, rope_cutoffs: Tuple[float, ...] = (256.0, 256.0, 256.0, 64.0)):
        super().__init__()
        self.attn = CrossAttentionRoPE4D(embed_dim=d_model, n_head=n_head, d_pos=4, rope_cutoffs=rope_cutoffs)

    @staticmethod
    def _build_coords_for_layer(tokens: torch.Tensor,  # [B,T,N,D]
                                hw: Optional[Tuple[int, int]],
                                layer_pos_norm: float) -> torch.Tensor:
        B, T, N, _ = tokens.shape
        device = tokens.device
        H, W = _infer_hw_from_n(N) if hw is None else hw
        assert H * W == N, f"H*W={H*W} must equal N={N}"
        t = torch.linspace(0.0, 1.0, T, device=device)
        y = torch.linspace(0.0, 1.0, H, device=device)
        x = torch.linspace(0.0, 1.0, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([
            t.view(T, 1, 1).expand(T, H, W),                            # t
            yy.expand(T, H, W),                                         # y
            xx.expand(T, H, W),                                         # x
            torch.full((T, H, W), float(layer_pos_norm), device=device) # l (layer rank in [0,1])
        ], dim=-1).view(T, N, 4)
        return coords.unsqueeze(0).expand(B, T, N, 4).contiguous()      # [B,T,N,4]

    def forward(
        self,
        features: List[torch.Tensor],      # each [T,N,D] or [B,T,N,D]
        chosen: Sequence[int],             # 1-based layer indices to include
        anchor_idx: int,                   # 1-based index from 'chosen' to act as queries
        hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        assert len(chosen) >= 1, "chosen must be non-empty"
        # Gather tensors and ensure [B,T,N,D]
        feats_btnd: List[torch.Tensor] = []
        added_batch_flags: List[bool] = []
        for k in chosen:
            f, added = _ensure_btnd(features[k - 1])
            feats_btnd.append(f)     # [B,T,N,D]
            added_batch_flags.append(added)
        # consistent batch across layers
        
        B = feats_btnd[0].shape[0]
        assert all(x.shape[0] == B for x in feats_btnd), "all layers must share the same batch size"

        # anchor
        anchor = feats_btnd[chosen.index(anchor_idx)]  # [B,T,Nq,D]
        B, T, Nq, D = anchor.shape

        # Q coords for anchor
        lq = _normalize_index_position(anchor_idx, list(chosen))
        coords_q = self._build_coords_for_layer(anchor, hw, lq)  # [B,T,Nq,4]

        # K/V by collapsing (layer, spatial) → Nk = L*N
        kv_list, ck_list = [], []
        for k in chosen:
            f = feats_btnd[chosen.index(k)]  # [B,T,N,D]
            lk = _normalize_index_position(k, list(chosen))
            ck = self._build_coords_for_layer(f, hw, lk)  # [B,T,N,4]
            kv_list.append(f)
            ck_list.append(ck)
        kv = torch.cat(kv_list, dim=2)       # [B,T,L*N,D]
        coords_k = torch.cat(ck_list, dim=2) # [B,T,L*N,4]
        # breakpoint()
        # RoPE cross-attn: anchor ← all chosen layers
        out = self.attn(q_tokens=anchor, kv_tokens=kv, coords_q=coords_q, coords_k=coords_k)  # [B,T,Nq,D]

        # squeeze batch if we added it for all inputs
        if all(added_batch_flags):
            out = out.squeeze(0)
        return out  # [T,N,D] or [B,T,N,D]





# ---------------------------
# Example with your shapes
# ---------------------------
if __name__ == "__main__":
    # DINO:   [1, 13, 49,  1024]
    # SigLIP: [1, 13, 144, 1152]
    B, T = 1, 13
    Nd, Ns = 49, 144
    Dd, Ds = 1024, 1152

    image_features2 = torch.randn(B, T, Nd, Dd)  # DINO
    image_features  = torch.randn(B, T, Ns, Ds)  # SigLIP

    model = SigLIP_DINO_CrossRoPE4D(
        d_siglip=Ds, d_dino=Dd,
        d_model=1024, n_head=8,
        cutoff_spatial_mask=0.25,   # optional hard mask in normalized coords
        dropout=0.0,
        attn_dist_mode="none",
        bidirectional=False,
        rope_cutoff_t=256.0,
        rope_cutoff_xy=256.0,
    )

    out = model(
        feat_siglip=image_features,
        feat_dino=image_features2,
        hw_siglip=(12, 12),
        hw_dino=(7, 7),
        use_time=True,
    )
    print("sig_fused:", out["sig_fused"].shape)  # -> [1, 13, 144, 1024]
