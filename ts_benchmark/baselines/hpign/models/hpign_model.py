import torch
import torch.nn as nn

from ..layers.embed import MultiScaleEmbedding
from ..layers.causal_block import CausalBlock
from ..layers.transformer import Encoder


class PredictionHead(nn.Module):
    def __init__(self, d_model, pred_len, series_dim):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len * series_dim)
        )

        self.pred_len = pred_len
        self.series_dim = series_dim

    def forward(self, x):
        # x: [B, D, d_model]
        B, D, _ = x.shape

        x = self.linear(x)
        x = x.view(B, D, self.pred_len)

        return x.permute(0, 2, 1)


class HPIGNModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.series_dim = config.series_dim

        self.alpha = config.alpha

        self.embedding = MultiScaleEmbedding(config)

        self.scale_encoders = nn.ModuleList([
            Encoder(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                num_layers=config.e_layers,
                dropout=config.dropout
            )
            for _ in range(config.wavelet_levels + 1)
        ])

        self.causal_blocks = nn.ModuleList([
            CausalBlock(config)
            for _ in range(config.wavelet_levels + 1)
        ])

        self.scale_heads = nn.ModuleList([
            PredictionHead(config.d_model, self.pred_len, self.series_dim)
            for _ in range(config.wavelet_levels + 1)
        ])

        self.fusion = nn.Sequential(
            nn.Linear(self.pred_len * (config.wavelet_levels + 1),
                      self.pred_len),
            nn.GELU(),
            nn.Linear(self.pred_len, self.pred_len)
        )

        self.global_proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x):
        """
        x: [B, L, enc_in]
        """

        embed_out = self.embedding(x)

        multi_scale = embed_out["multi_scale"]
        x_global = embed_out["x_global"]       # [B, D, d_model]
        exog_global = embed_out["exog_global"] # [B, D, d_model]

        global_context = self.global_proj(x_global.mean(dim=1))  # [B, d_model]

        scale_preds = []
        te_list = []
        attn_list = []
        recon_inputs = []

        for i, ((x_patch, e_patch), encoder, block, head) in enumerate(
                zip(multi_scale, self.scale_encoders, self.causal_blocks, self.scale_heads)):

            # x_patch: [B, D, d_model, N]
            B, D, C, N = x_patch.shape

            # ===== reshape =====
            x_patch = x_patch.reshape(B * D, C, N).permute(0, 2, 1)
            e_patch = e_patch.reshape(B * D, C, N).permute(0, 2, 1)


            x_patch = encoder(x_patch)
            e_patch = encoder(e_patch)

 
            out, te_matrix, attn = block(
                x_patch,
                e_patch,
                x_global,
                exog_global,
                global_context=global_context  
            )


            out = out.mean(dim=1)
            out = out.view(B, D, -1)


            pred = head(out)

            scale_preds.append(pred)
            te_list.append(te_matrix)
            attn_list.append(attn)

            recon_inputs.append(out)


        stacked = torch.cat(
            [p.reshape(p.shape[0], -1) for p in scale_preds],
            dim=-1
        )

        fused = self.fusion(stacked)
        fused = fused.unsqueeze(-1)


        recon = self.embedding.reconstruct(scale_preds)

        return {
            "pred": fused,
            "multi_scale_preds": scale_preds,
            "reconstruction": recon,
            "te_matrices": te_list,
            "attentions": attn_list,
            "raw": embed_out["raw"]
        }