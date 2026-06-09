import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride

        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_len,
            stride=stride,
            padding=padding
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, D, L]
        B, D, L = x.shape

        x = x.reshape(B * D, 1, L)
        x = self.proj(x)   # [B*D, d_model, N_patch]

        _, C, N = x.shape
        x = x.reshape(B, D, C, N)

        return self.dropout(x), D



class SeriesProjector(nn.Module):
    def __init__(self, input_dim, seq_len, d_model):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x.permute(0, 2, 1)  # [B, D, L]
        x = self.proj(x)        # [B, D, d_model]
        x = x.mean(dim=1)       # [B, d_model]
        return x



class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, mode):
        if mode == "norm":
            self.mean = x.mean(dim=1, keepdim=True).detach()
            self.std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.std

        elif mode == "denorm":
            return x * self.std + self.mean



class MultiScaleDecomposition(nn.Module):
    def __init__(self, levels=2):
        super().__init__()
        self.levels = levels

    def forward(self, x):
        """
        x: [B, L, D]
        return: list of multi-scale components
        """

        outputs = []
        current = x

        for _ in range(self.levels):
            low = F.avg_pool1d(current.permute(0, 2, 1), kernel_size=2, stride=2)
            up = F.interpolate(low, size=current.shape[1], mode="linear", align_corners=False)
            high = current.permute(0, 2, 1) - up

            low = low.permute(0, 2, 1)
            high = high.permute(0, 2, 1)

            outputs.append(high)
            current = low

        outputs.append(current)  # final low freq

        return outputs



class MultiScaleReconstruction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, components):
        """
        components: list of [B, L, D]
        """
        out = 0
        for comp in components:
            if comp.shape[1] != components[0].shape[1]:
                comp = F.interpolate(comp.permute(0, 2, 1),
                                     size=components[0].shape[1],
                                     mode="linear",
                                     align_corners=False).permute(0, 2, 1)
            out = out + comp
        return out



class MultiScaleEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.series_dim = config.series_dim
        self.d_model = config.d_model
        self.patch_len = config.patch_len
        self.stride = config.stride

        self.use_wavelet = getattr(config, "use_wavelet", True)
        self.levels = getattr(config, "wavelet_levels", 2)

        self.revin_x = RevIN()
        self.revin_exog = RevIN()

        self.decomposition = MultiScaleDecomposition(self.levels)
        self.reconstruction = MultiScaleReconstruction()

        self.x_patch = PatchEmbedding(
            self.d_model,
            self.patch_len,
            self.stride,
            padding=self.stride,
            dropout=config.dropout
        )

        self.exog_patch = PatchEmbedding(
            self.d_model,
            self.patch_len,
            self.stride,
            padding=self.stride,
            dropout=config.dropout
        )

        self.x_proj = SeriesProjector(self.series_dim, config.seq_len, self.d_model)
        self.exog_proj = SeriesProjector(config.enc_in - self.series_dim, config.seq_len, self.d_model)

    def forward(self, x):
        """
        x: [B, L, enc_in]
        """

        x_endo = x[:, :, :self.series_dim]
        x_exog = x[:, :, self.series_dim:]

        x_endo = self.revin_x(x_endo, "norm")
        x_exog = self.revin_exog(x_exog, "norm")

        if self.use_wavelet:
            x_scales = self.decomposition(x_endo)
            exog_scales = self.decomposition(x_exog)
        else:
            x_scales = [x_endo]
            exog_scales = [x_exog]

        embedded_scales = []

        for xs, es in zip(x_scales, exog_scales):
            xs = xs.permute(0, 2, 1)
            es = es.permute(0, 2, 1)

            x_patch, _ = self.x_patch(xs)
            e_patch, _ = self.exog_patch(es)

            embedded_scales.append((x_patch, e_patch))

        x_global = self.x_proj(x_endo)
        exog_global = self.exog_proj(x_exog)

        return {
            "multi_scale": embedded_scales,
            "x_global": x_global,
            "exog_global": exog_global,
            "raw": (x_endo, x_exog)
        }

    def reconstruct(self, outputs):
        return self.reconstruction(outputs)