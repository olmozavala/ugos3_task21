import math

import torch
import torch.nn.functional as F


_SOBEL_X_M = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], requires_grad=False).view(1, 1, 3, 3)
_SOBEL_Y_M = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], requires_grad=False).view(1, 1, 3, 3)


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply 2D Gaussian blur. Input: [B,H,W] or [B,1,H,W]. Output: same shape."""
    squeeze = x.dim() == 3
    if squeeze:
        x = x.unsqueeze(1)
    k = 2 * math.ceil(3 * sigma) + 1
    ax = torch.arange(-(k // 2), k // 2 + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, k, k)
    out = F.conv2d(x, kernel_2d, padding=k // 2)
    return out.squeeze(1) if squeeze else out


def masked_rmse(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """RMSE over valid pixels only. mask=1 means valid."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=pred.dtype, device=pred.device)
    valid = mask.sum() + eps
    mse = (((pred - target) ** 2) * mask).sum() / valid
    return torch.sqrt(mse)


def gradient_rmse(pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """RMSE on Sobel gradient magnitudes over valid pixels."""
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(dtype=pred.dtype, device=pred.device)
    kx = _SOBEL_X_M.to(device=pred.device, dtype=pred.dtype)
    ky = _SOBEL_Y_M.to(device=pred.device, dtype=pred.dtype)
    gx_pred = F.conv2d(pred, kx, padding=1)
    gy_pred = F.conv2d(pred, ky, padding=1)
    gx_tgt = F.conv2d(target, kx, padding=1)
    gy_tgt = F.conv2d(target, ky, padding=1)
    mag_pred = torch.sqrt(gx_pred ** 2 + gy_pred ** 2)
    mag_tgt = torch.sqrt(gx_tgt ** 2 + gy_tgt ** 2)
    valid = mask.sum() + eps
    mse = (((mag_pred - mag_tgt) ** 2) * mask).sum() / valid
    return torch.sqrt(mse)


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
