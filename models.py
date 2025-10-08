import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv3d → BatchNorm3d → ReLU) × 2"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpSampleConv(nn.Module):
    """Upsample by 2× with interpolate, then DoubleConv on concatenated features."""

    def __init__(self, in_ch, out_ch, mode="trilinear", align_corners=False):
        super().__init__()
        # in_ch = channels from lower-res + skip channels
        self.double_conv = DoubleConv(in_ch, out_ch)
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x, skip):
        # 1) Upsample
        x = F.interpolate(
            x, scale_factor=2, mode=self.mode, align_corners=self.align_corners
        )
        # 2) If needed, pad to match skip size (odd dimensions)
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
        # 3) Concatenate and convolve
        x = torch.cat([skip, x], dim=1)
        return self.double_conv(x)


class UNet3D(nn.Module):
    """3D U-Net architecture for image to image mappings"""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=[16, 32, 64],
    ):
        super().__init__()

        # Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv(prev_ch, feat))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_ch = feat

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder (using interpolate + conv)
        self.ups = nn.ModuleList()
        for feat in reversed(features):
            # in channels = feat*2 (from bottleneck or prev up) + feat (skip)
            self.ups.append(UpSampleConv(feat * 2 + feat, feat))

        # Final 1×1×1 conv
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # PET images can have arbitrary global scales, but we don't want to
        # normalize before calculating the log-likelihood gradient
        # instead we calculate scales of all images in the batch and apply
        # them only before using the neural network

        # as scale we use the mean of the input images
        # if we are using early stopped OSEM images, the mean is well defined

        input_x = x  # Save the original input

        skips = []
        # Encoder path
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        unet_out = self.final_conv(x)

        return input_x + unet_out
