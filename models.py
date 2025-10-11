from typing import List, Tuple, Optional
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
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mode: str = "trilinear",
        align_corners: Optional[bool] = False,
    ):
        super().__init__()
        self.double_conv = DoubleConv(in_ch, out_ch)
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # TorchScript-friendly interpolate
        sz = [int(s * 2) for s in x.shape[2:]]
        x = F.interpolate(
            x,
            size=sz,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=False,
        )
        # pad if needed
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
        start_features: int = 16,
        num_levels: int = 3,
        down_conv: bool = True,
        final_softplus: bool = False,
    ):
        super().__init__()

        self.features = [start_features * (2**i) for i in range(num_levels)]

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for feat in self.features:
            self.downs.append(DoubleConv(prev_ch, feat))
            if down_conv:
                self.pools.append(
                    nn.Conv3d(
                        feat, feat, kernel_size=3, stride=2, padding=1, bias=False
                    )
                )
            else:
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_ch = feat

        self.bottleneck = DoubleConv(self.features[-1], self.features[-1] * 2)

        if final_softplus:
            self.final_act = nn.Softplus(beta=5.0)
        else:
            self.final_act = nn.ReLU()

        self.ups = nn.ModuleList()
        for feat in list(reversed(self.features)):
            self.ups.append(UpSampleConv(feat * 2 + feat, feat))

        self.final_conv = nn.Conv3d(self.features[0], out_channels, kernel_size=1)
        self.donv_conv = down_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_x = x
        skips: List[torch.Tensor] = []

        # Encoder — iterate modules directly (no integer indexing)
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder — enumerate ups; index only into the *tensor* list (ok)
        for idx, up in enumerate(self.ups):
            skip = skips[len(skips) - 1 - idx]
            x = up(x, skip)

        # add final Relu to guarantee non-negative output
        return self.final_act(input_x + self.final_conv(x))
