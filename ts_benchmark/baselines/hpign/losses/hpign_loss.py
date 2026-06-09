import torch
import torch.nn as nn
import torch.nn.functional as F


class HPIGNLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pred_loss_type = config.loss
        self.alpha = config.alpha          # causality
        self.beta = config.beta            # multi-scale
        self.gamma = getattr(config, "gamma", 0.1)  # reconstruction
        self.delta = getattr(config, "delta", 0.05) # regularization

        if self.pred_loss_type == "MSE":
            self.pred_loss_fn = nn.MSELoss()
        elif self.pred_loss_type == "MAE":
            self.pred_loss_fn = nn.L1Loss()
        else:
            self.pred_loss_fn = nn.HuberLoss(delta=0.5)

    def forward(self, outputs, target):
        """
        outputs dict:
            {
                "pred": final prediction
                "multi_scale_preds": list
                "reconstruction": reconstructed signal
                "te_matrices": list
                "attentions": list
            }
        """

        pred = outputs["pred"]


        pred_loss = self.pred_loss_fn(pred, target)


        ms_loss = 0
        if "multi_scale_preds" in outputs:
            ms_preds = outputs["multi_scale_preds"]

            for i in range(len(ms_preds)):
                for j in range(i + 1, len(ms_preds)):
                    ms_loss += F.mse_loss(ms_preds[i], ms_preds[j])

            ms_loss = ms_loss / (len(ms_preds) + 1e-6)


        causality_loss = 0
        if "te_matrices" in outputs and "attentions" in outputs:
            te_list = outputs["te_matrices"]
            attn_list = outputs["attentions"]

            for te, attn in zip(te_list, attn_list):
                # normalize TE
                te_norm = te / (te.mean() + 1e-6)

                # attention mean over heads
                attn_mean = attn.mean(dim=1)

                causality_loss += F.mse_loss(attn_mean, te_norm)

            causality_loss = causality_loss / (len(te_list) + 1e-6)


        recon_loss = 0
        if "reconstruction" in outputs and "raw" in outputs:
            recon = outputs["reconstruction"]
            raw_x, _ = outputs["raw"]

            recon_loss = F.mse_loss(recon, raw_x)


        reg_loss = 0
        if "attentions" in outputs:
            for attn in outputs["attentions"]:
                # sparsity (encourage causal selectivity)
                reg_loss += torch.mean(torch.abs(attn))

                # entropy regularization (avoid collapse)
                prob = F.softmax(attn, dim=-1)
                reg_loss += torch.mean(prob * torch.log(prob + 1e-6))


        total_loss = (
            pred_loss
            + self.alpha * causality_loss
            + self.beta * ms_loss
            + self.gamma * recon_loss
            + self.delta * reg_loss
        )

        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "causality_loss": causality_loss,
            "ms_loss": ms_loss,
            "recon_loss": recon_loss,
            "reg_loss": reg_loss,
        }