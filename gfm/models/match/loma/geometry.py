import torch
from .device import device

def get_normalized_grid(
    B: int,
    H: int,
    W: int,
    overload_device: torch.device | None = None,
) -> torch.Tensor:
    if B == 0:
        return torch.zeros(0, H, W, 2, device=overload_device or device)
    x1_n = torch.meshgrid(
        *[
            torch.linspace(-1 + 1 / n, 1 - 1 / n, n, device=overload_device or device)
            for n in (B, H, W)
        ],
        indexing="ij",
    )
    x1_n = torch.stack((x1_n[2], x1_n[1]), dim=-1).reshape(B, H, W, 2)
    return x1_n


def kde(x: torch.Tensor, std: float = 0.1, half: bool = True) -> torch.Tensor:
    # use a gaussian kernel to estimate density
    if half:
        x = x.half()
    scores = (-(torch.cdist(x, x) ** 2) / (2 * std**2)).exp()
    density = scores.sum(dim=-1)
    return density
