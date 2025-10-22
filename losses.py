import torch
from ignite.metrics import SSIM


class RobustL1Loss(torch.nn.Module):
    """
    Robust L1 / Charbonnier loss: sqrt((x - y)^2 + eps^2)

    Args:
        eps (float): small constant for numerical stability (typical 1e-3 ~ 1e-6).
        reduction (str): 'mean' | 'sum' | 'none' | 'batchmean'
            - 'mean': mean over all elements in the batch
            - 'sum': sum over all elements
            - 'none': per-element loss (same shape as input)
            - 'batchmean': mean per-sample (over non-batch dims), then mean over batch
    """

    def __init__(self, eps: float = 1e-2, reduction: str = "mean"):
        super().__init__()
        assert reduction in {"mean", "sum", "none", "batchmean"}
        self.eps = eps
        self.reduction = reduction

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)

        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "batchmean":
            # average over non-batch dims, then mean across batch
            if loss.ndim == 1:
                per_sample = loss
            else:
                per_sample = loss.view(loss.shape[0], -1).mean(dim=1)
            return per_sample.mean()


# --- Simple 3D edge loss (finite differences) ---
def grad_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    def diffs(t):
        dx = t[:, :, 1:, :, :] - t[:, :, :-1, :, :]
        dy = t[:, :, :, 1:, :] - t[:, :, :, :-1, :]
        dz = t[:, :, :, :, 1:] - t[:, :, :, :, :-1]
        return dx, dy, dz

    dx_x, dy_x, dz_x = diffs(x)
    dx_y, dy_y, dz_y = diffs(y)

    return (
        (dx_x - dx_y).abs().mean()
        + (dy_x - dy_y).abs().mean()
        + (dz_x - dz_y).abs().mean()
    )


class L1SSIMEdgeLoss(torch.nn.Module):
    """
    Combines: Robust L1 (Charbonnier), 3D SSIM (Ignite), and edge loss.
    Expects x_hat, y in [0, 3.5] with shape [B, C, D, H, W].
    """

    def __init__(self, w_charb: float = 0.5, w_ssim: float = 0.4):
        super().__init__()
        self.w_charb = w_charb
        self.w_ssim = w_ssim
        self.w_edge = 1 - self.w_charb - self.w_ssim

        self.charb = RobustL1Loss(eps=1e-2, reduction="mean")

        # Ignite SSIM is stateful; create once and reuse
        self.ssim3d = SSIM(
            data_range=3.5,
            kernel_size=11,
            sigma=1.5,
            k1=0.01,
            k2=0.03,
            ndims=3,  # needs pytorch-ignite >= 0.5.2
        )

    @torch.no_grad()
    def _ssim3d_value(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.ssim3d.to(x.device)
        self.ssim3d.reset()
        self.ssim3d.update((x, y))
        val = self.ssim3d.compute()
        return torch.as_tensor(val, device=x.device, dtype=x.dtype)

    def forward(self, x_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        l_charb = self.charb(x_hat, y)
        l_ssim = 1.0 - self._ssim3d_value(x_hat, y)
        l_edge = grad_l1(x_hat, y)
        return self.w_charb * l_charb + self.w_ssim * l_ssim + self.w_edge * l_edge


l1_ssim_edge_loss_1 = L1SSIMEdgeLoss(w_charb=0.7, w_ssim=0.25)
l1_ssim_edge_loss_2 = L1SSIMEdgeLoss(w_charb=0.5, w_ssim=0.4)
