import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None, attn_bias=None):
        d_k = Q.size(-1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)

        if attn_bias is not None:
            scores = scores + attn_bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V)
        return output, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape

        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        out, attn = self.attn(Q, K, V, mask)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        return out, attn


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(dropout)

    def forward(self, query, key, value, mask=None, attn_bias=None):
        B, Lq, D = query.shape
        Lk = key.shape[1]

        Q = self.q_proj(query).view(B, Lq, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(B, Lk, self.n_heads, self.d_head).transpose(1, 2)

        out, attn = self.attn(Q, K, V, mask, attn_bias)

        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.out_proj(out)

        return out, attn



class TransferEntropyAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.cross_attn = MultiHeadCrossAttention(d_model, n_heads, dropout)

        self.te_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_heads)
        )

    def forward(self, x, exog, te_matrix):
        """
        x: endogenous features (query)
        exog: exogenous features (key/value)
        te_matrix: [B, D, D] or [B, 1] simplified TE signal
        """

        B, L, D = x.shape

        # reduce TE to scalar per batch (or could be channel-wise)
        if te_matrix.dim() == 3:
            te_scalar = te_matrix.mean(dim=(1, 2), keepdim=True)  # [B,1,1]
        else:
            te_scalar = te_matrix

        te_weight = self.te_proj(te_scalar)  # [B, n_heads]
        te_weight = te_weight.unsqueeze(-1).unsqueeze(-1)  # [B, H, 1, 1]

        out, attn = self.cross_attn(x, exog, exog)

        # modulate attention output
        out = out * (1 + te_weight.mean())

        return out, attn



class HybridCausalAttention(nn.Module):
    """
    Combine:
    - Self-attention (endogenous)
    - TE-guided cross-attention (exogenous → endogenous)
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = TransferEntropyAttention(d_model, n_heads, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, exog, te_matrix):
        # self-attention
        x_self, _ = self.self_attn(x)
        x = x + self.dropout(x_self)
        x = self.norm1(x)

        # TE-guided cross attention
        x_cross, _ = self.cross_attn(x, exog, te_matrix)
        x = x + self.dropout(x_cross)
        x = self.norm2(x)

        return x