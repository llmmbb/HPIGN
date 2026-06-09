import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.scale = (d_model // n_heads) ** -0.5

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.out = nn.Linear(d_model, d_model)

        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def compute_entropy(self, attn):
        return -(attn * torch.log(attn + 1e-6)).sum(-1)

    def forward(
        self,
        x,
        context=None,
        attn_bias=None,
        attn_alpha=None
    ):
        """
        x: [B, N, D]
        context: for cross attention
        attn_bias: causal prior (TE)
        attn_alpha: dynamic scaling
        """

        if context is None:
            context = x

        Q = self.q(x)
        K = self.k(context)
        V = self.v(context)

        B, N, D = Q.shape

        Q = Q.view(B, N, self.n_heads, D // self.n_heads).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, D // self.n_heads).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, D // self.n_heads).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if attn_bias is not None:
            attn = attn + attn_bias

        attn = torch.softmax(attn, dim=-1)

        entropy = self.compute_entropy(attn)

        gate = torch.sigmoid(self.gate(x)).unsqueeze(1)
        attn = attn * gate

        if attn_alpha is not None:
            attn = attn * attn_alpha

        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out(out)

        return out, entropy



class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout):
        super().__init__()

        self.self_attn = CausalAttention(d_model, n_heads)
        self.cross_attn = CausalAttention(d_model, n_heads)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x,
        cross=None,
        attn_bias=None,
        attn_alpha=None
    ):
        out, ent1 = self.self_attn(
            x,
            context=None,
            attn_bias=None,
            attn_alpha=None
        )
        x = self.norm1(x + self.dropout(out))

        if cross is not None:
            out, ent2 = self.cross_attn(
                x,
                context=cross,
                attn_bias=attn_bias,
                attn_alpha=attn_alpha
            )
            x = self.norm2(x + self.dropout(out))
        else:
            ent2 = 0

        ff = self.ffn(x)
        x = self.norm3(x + self.dropout(ff))

        return x, ent1, ent2



class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x,
        cross=None,
        attn_bias=None,
        attn_alpha=None
    ):
        entropy_list = []

        for layer in self.layers:
            x, ent1, ent2 = layer(
                x,
                cross=cross,
                attn_bias=attn_bias,
                attn_alpha=attn_alpha
            )

            entropy_list.append(ent1)
            if isinstance(ent2, torch.Tensor):
                entropy_list.append(ent2)

        return x, entropy_list