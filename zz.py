from typing import List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpSampleConv(nn.Module):
    """Upsample (interpolate) -> concat skip -> refine with DoubleConv."""

    def __init__(
        self,
        in_ch_after_cat: int,  # channels AFTER concatenation (skip || upsampled)
        out_ch: int,
        mode: str = "trilinear",
        align_corners: Optional[bool] = False,
    ):
        super().__init__()
        self.double_conv = DoubleConv(in_ch_after_cat, out_ch)
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # TorchScript-friendly size computation (avoid scale_factor)
        sz = [int(s * 2) for s in x.shape[2:]]
        x = F.interpolate(
            x,
            size=sz,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=False,
        )
        # pad if needed (handles odd sizes)
        if x.shape[2:] != skip.shape[2:]:
            diffZ = skip.size(2) - x.size(2)
            diffY = skip.size(3) - x.size(3)
            diffX = skip.size(4) - x.size(4)
            x = F.pad(
                x,
                [
                    diffX // 2,
                    diffX - diffX // 2,
                    diffY // 2,
                    diffY - diffY // 2,
                    diffZ // 2,
                    diffZ - diffZ // 2,
                ],
            )
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: List[int] = [16, 32, 64],
        down_mode: str = "conv",  # "conv" (stride-2 conv) or "pool"
        out_activation: str = "softplus",  # "softplus" | "relu" | "identity"
        softplus_beta: float = 5.0,  # β for softplus
        softplus_floor: float = 0.01,  # ε: desired floor at zero input
    ):
        super().__init__()
        assert down_mode in ("conv", "pool")
        assert out_activation in ("softplus", "relu", "identity")
        self.out_activation = out_activation
        self.softplus_beta = float(softplus_beta)
        self.softplus_floor = float(softplus_floor)

        # Encoder
        self.downs = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        prev_ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv(prev_ch, feat))
            if down_mode == "pool":
                self.downsamplers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            else:
                # learned downsampling (often nicer for denoising than hard pooling)
                self.downsamplers.append(
                    nn.Conv3d(
                        feat, feat, kernel_size=3, stride=2, padding=1, bias=False
                    )
                )
            prev_ch = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder: derive concat channels explicitly (robust to any `features`)
        self.ups = nn.ModuleList()
        dec_in_ch = features[-1] * 2
        for skip_ch in reversed(features):
            in_after_cat = dec_in_ch + skip_ch
            self.ups.append(UpSampleConv(in_after_cat, skip_ch))
            dec_in_ch = skip_ch

        # 1x1 head predicts residual noise r(x)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1, bias=True)

        # Residual-friendly init: weights=0 (identity start), bias sets softplus floor
        nn.init.zeros_(self.final_conv.weight)
        if self.final_conv.bias is not None:
            eps = max(1e-8, self.softplus_floor)
            beta = self.softplus_beta
            b = (1.0 / beta) * math.log(math.expm1(beta * eps))
            self.final_conv.bias.data.fill_(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        skips: List[torch.Tensor] = []

        # Encoder
        for enc, down in zip(self.downs, self.downsamplers):
            x = enc(x)
            skips.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for idx, up in enumerate(self.ups):
            skip = skips[(len(skips) - 1) - idx]
            x = up(x, skip)

        y = inp - self.final_conv(x)

        # Output activation
        if self.out_activation == "softplus":
            return F.softplus(y, beta=self.softplus_beta)
        elif self.out_activation == "relu":
            return F.relu(y)
        else:
            return y
