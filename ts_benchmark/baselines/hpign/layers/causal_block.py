import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FullAttention, AttentionLayer


class TransferEntropyEstimator(nn.Module):
    def __init__(self, d_model, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        # x: [B, L, D] (endogenous)
        # y: [B, L, D] (exogenous)
        B, L, D = x.shape

        xy = torch.cat([x, y], dim=-1)
        score = self.net(xy)  # [B, L, 1]

        # global TE score
        te = score.mean(dim=1)  # [B, 1]
        return te


class DynamicTauGenerator(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # x: [B, L, D]
        tau = self.mlp(x.mean(dim=1))  # [B, 1]
        tau = torch.sigmoid(tau)
        return tau


class CausalAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout=0.1,
        activation="gelu",
        use_exog=True
    ):
        super().__init__()

        self.use_exog = use_exog

        # ===== Self Attention (Endogenous) =====
        self.self_attn = AttentionLayer(
            FullAttention(False, attention_dropout=dropout),
            d_model,
            n_heads
        )

        # ===== Cross Attention (Exogenous → Endogenous) =====
        if use_exog:
            self.cross_attn = AttentionLayer(
                FullAttention(False, attention_dropout=dropout),
                d_model,
                n_heads
            )

        # ===== Transfer Entropy =====
        if use_exog:
            self.te_estimator = TransferEntropyEstimator(d_model)

        # ===== Dynamic Tau =====
        self.tau_net = DynamicTauGenerator(d_model)

        # ===== Feed Forward =====
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # ===== Norms =====
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, exog=None):
        """
        x:     [B, L, D]  (endogenous)
        exog:  [B, L, D]  (exogenous)
        """

        # ===============================
        # 1. Self-Attention (Endogenous)
        # ===============================
        res = x
        x_self, attn_self = self.self_attn(x, x, x)
        x = self.norm1(res + self.dropout(x_self))

        causality_loss = 0.0
        te_value = None
        tau = None

        # ===============================
        # 2. Cross-Attention (TE-guided)
        # ===============================
        if self.use_exog and exog is not None:

            # ---- Transfer Entropy ----
            te_value = self.te_estimator(x, exog)  # [B, 1]

            # ---- Dynamic Tau ----
            tau = self.tau_net(x)  # [B, 1]

            # ---- Build Attention Bias ----
            # stronger TE → stronger cross attention
            attn_bias = te_value.unsqueeze(-1).unsqueeze(-1)  # [B,1,1,1]

            # ---- Cross Attention ----
            res = x
            x_cross, attn_cross = self.cross_attn(
                x, exog, exog,
                attn_bias=attn_bias,
                tau=tau
            )

            # ---- Gated Fusion ----
            gate = torch.sigmoid(te_value).unsqueeze(-1)  # [B,1,1]
            x = res + gate * x_cross
            x = self.norm2(x)

            # ---- Causality Loss ----
            causality_loss = torch.mean((1 - te_value) ** 2)

        # ===============================
        # 3. Feed Forward
        # ===============================
        res = x
        x_ffn = self.ffn(x)
        x = self.norm3(res + self.dropout(x_ffn))

        return x, causality_loss, te_value, tau


class CausalBlock(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        n_heads,
        d_ff,
        dropout=0.1,
        activation="gelu",
        use_exog=True
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            CausalAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                use_exog=use_exog
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, exog=None):
        total_loss = 0.0
        te_list = []
        tau_list = []

        for layer in self.layers:
            x, loss, te, tau = layer(x, exog)
            total_loss += loss

            if te is not None:
                te_list.append(te)
            if tau is not None:
                tau_list.append(tau)

        if len(te_list) > 0:
            te_stack = torch.stack(te_list, dim=0)  # [L, B, 1]
            tau_stack = torch.stack(tau_list, dim=0)
        else:
            te_stack, tau_stack = None, None

        return x, total_loss, te_stack, tau_stack