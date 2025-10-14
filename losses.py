import torch


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
