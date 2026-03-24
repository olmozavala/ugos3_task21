"""
Loss functions for DA_Chlora.

This module supports:
- simple single-loss selection via config: `"loss": "mse_loss"`
- weighted combinations via config, e.g.

  "loss": [
    {"name": "masked_mse_loss", "weight": 1.0},
    {"name": "sobel_gradient_magnitude_loss", "weight": 0.9, "args": {"normalize": true}},
    {"name": "laplacian_curvature_loss", "weight": 0.5}
  ]

All losses are callable as:
    loss(output, target, data=None, mask=None, **kwargs)

If `mask` is not provided and `data` is provided, the mask is extracted from
`data[:, -1, :, :]` (last input channel).

### For the NEMO dataset, between (2016-04-01, 2021-07-04) ###
#  Variance of first derivatives: 0.15808774535211767        #
#  Variance of second derivatives: 0.008478596971365743      #
##############################################################
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 3)
_LAPLACE = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32, requires_grad=False).view(1, 1, 3, 3)

_VAR_GRAD = 0.15808774535211767
_VAR_CURV = 0.008478596971365743


def _to_b1hw(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize input tensor to [B,1,H,W] when possible.
    Accepts:
    - [H,W] -> [1,1,H,W]
    - [B,H,W] -> [B,1,H,W]
    - [B,1,H,W] -> unchanged
    """
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 3:
        return x.unsqueeze(1)
    if x.dim() == 4:
        return x
    raise ValueError(f"Expected 2D/3D/4D tensor, got shape {tuple(x.shape)}")


def _extract_mask(mask: Optional[torch.Tensor], data: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is not None:
        return mask
    if data is None:
        return None
    if data.dim() != 4:
        raise ValueError(f"Cannot extract mask from data with shape {tuple(data.shape)}; expected [B,C,H,W]")
    # last channel is the mask
    return data[:, -1, :, :].unsqueeze(1)


def _valid_points_from_mask(mask_b1hw: Optional[torch.Tensor], shape_like: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (mask_b1hw, valid_points) where mask is always [B,1,H,W].
    If mask is None, use all-ones mask.
    """
    if mask_b1hw is None:
        ones = torch.ones_like(shape_like, dtype=shape_like.dtype, device=shape_like.device)
        return ones, torch.tensor(float(ones.numel()), device=shape_like.device, dtype=shape_like.dtype)
    mask_b1hw = _to_b1hw(mask_b1hw).to(device=shape_like.device, dtype=shape_like.dtype)
    valid_points = mask_b1hw.sum()
    # prevent division by zero
    valid_points = valid_points + torch.tensor(eps, device=shape_like.device, dtype=shape_like.dtype)
    return mask_b1hw, valid_points


def _filter_kwargs(fn: Callable[..., Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Filter kwargs to only those accepted by fn, unless fn has **kwargs.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return dict(kwargs)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def _resolve_loss_callable(name: str, args: Optional[Mapping[str, Any]] = None) -> Callable[..., torch.Tensor]:
    """
    Resolve a loss by name from this module.
    - If it's a function: returns it.
    - If it's an nn.Module class: instantiates it.
    - If it's an nn.Module instance: returns it.
    The resulting callable is wrapped to accept extra kwargs safely.
    """
    module = sys.modules[__name__]
    if not hasattr(module, name):
        raise AttributeError(f"Unknown loss '{name}' in model/loss.py")
    obj = getattr(module, name)
    fixed_args = dict(args or {})

    if isinstance(obj, torch.nn.Module):
        loss_obj = obj
        forward = loss_obj.forward
        def _call(output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return forward(output, target, **_filter_kwargs(forward, kwargs))
        return _call

    if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
        loss_obj = obj(**fixed_args)
        forward = loss_obj.forward
        def _call(output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return forward(output, target, **_filter_kwargs(forward, kwargs))
        return _call

    if callable(obj):
        def _call(output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            merged = dict(fixed_args)
            merged.update(kwargs)  # runtime kwargs override config-fixed args
            call_kwargs = _filter_kwargs(obj, merged)
            return obj(output, target, **call_kwargs)
        return _call

    raise TypeError(f"Loss '{name}' is not callable")


def build_loss(loss_spec: Any) -> Callable[..., torch.Tensor]:
    """
    Build a criterion from `config['loss']`.

    Supported formats:
    - string: "mse_loss"
    - list: [{"name": "...", "weight": 1.0, "args": {...}}, ...]
    - dict: {"type": "weighted_sum", "args": {...}}  (see weighted_sum_loss)
    """
    if isinstance(loss_spec, str):
        return _resolve_loss_callable(loss_spec)

    if isinstance(loss_spec, list):
        return WeightedSumLoss(loss_spec)

    if isinstance(loss_spec, dict):
        loss_type = loss_spec.get("type")
        args = dict(loss_spec.get("args", {}))
        if loss_type in ("weighted_sum", "weighted_sum_loss", "WeightedSum"):
            terms = args.get("terms") or args.get("losses")
            if not isinstance(terms, list):
                raise ValueError("weighted_sum loss requires args.terms (or args.losses) to be a list")
            normalize = bool(args.get("normalize", True))
            eps = float(args.get("eps", 1e-10))
            return WeightedSumLoss(terms, normalize=normalize, eps=eps)
        # allow dict specifying a single loss with args:
        if isinstance(loss_type, str):
            return _resolve_loss_callable(loss_type, args)
        raise ValueError(f"Unsupported loss dict spec: {loss_spec}")

    raise TypeError(f"Unsupported loss spec type: {type(loss_spec)}")


# ----------------------------
# Base / simple losses
# ----------------------------

def nll_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.nll_loss(output, target)


def mse_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Keep this generic, but align common [B,H,W] vs [B,1,H,W] mismatch.
    if output.dim() == 3 and target.dim() == 4 and target.size(1) == 1:
        output = output.unsqueeze(1)
    elif output.dim() == 4 and output.size(1) == 1 and target.dim() == 3:
        target = target.unsqueeze(1)
    return F.mse_loss(output, target)


def mae_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Keep this generic, but align common [B,H,W] vs [B,1,H,W] mismatch.
    if output.dim() == 3 and target.dim() == 4 and target.size(1) == 1:
        output = output.unsqueeze(1)
    elif output.dim() == 4 and output.size(1) == 1 and target.dim() == 3:
        target = target.unsqueeze(1)
    return F.l1_loss(output, target)


def mse_loss_with_mask(output: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    output, target = _to_b1hw(output), _to_b1hw(target)
    mask_b1hw, valid = _valid_points_from_mask(mask, output, eps=1e-10)
    return (((output - target) ** 2) * mask_b1hw).sum() / valid

def total_variation_loss(output: torch.Tensor, 
                         target: torch.Tensor, 
                         data: Optional[torch.Tensor] = None, 
                         mask: Optional[torch.Tensor] = None,
                         regularization: str = "l1",
                         eps: float = 1e-10) -> torch.Tensor:
    """
    Total variation loss (masked, mean over valid neighbor pairs).

    Notes:
    - `target` is unused (TV is a regularizer on `output`).
    - `regularization` is the type of regularization to use. Options are: 'l1' or 'l2'

    - If a mask is available, we only count differences where BOTH neighboring
      pixels are valid.
    """
    output, target = _to_b1hw(output), _to_b1hw(target)
    mask = _extract_mask(mask, data)
    mask_b1hw, _ = _valid_points_from_mask(mask, output, eps=eps)

    # this is VERTICAL (along height)
    diff_h = output[..., 1:, :] - output[..., :-1, :]
    mask_h = mask_b1hw[..., 1:, :] * mask_b1hw[..., :-1, :]

    # this is HORIZONTAL (along width)
    diff_w = output[..., 1:] - output[..., :-1]
    mask_w = mask_b1hw[..., 1:] * mask_b1hw[..., :-1]
    if regularization == "l1":
        tv_h = (diff_h.abs() * mask_h).sum()
        tv_w = (diff_w.abs() * mask_w).sum()
    elif regularization == "l2":
        tv_h = (diff_h.square() * mask_h).sum()
        tv_w = (diff_w.square() * mask_w).sum()
    else:
       raise ValueError(f"Invalid regularization type: {regularization!r}. Expected 'l1' or 'l2'")

    denom = mask_h.sum() + mask_w.sum() + output.new_tensor(eps)
    return (tv_h + tv_w) / denom


# ----------------------------
# Mask-aware terms used in Trainer (pixel / gradient / curvature)
# ----------------------------

def masked_mse_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    data: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Pixelwise MSE reduced by mask (mean over valid points).
    Matches the previous in-trainer `output_loss`.
    """
    output, target = _to_b1hw(output), _to_b1hw(target)
    mask = _extract_mask(mask, data)
    mask_b1hw, valid = _valid_points_from_mask(mask, output, eps=eps)
    return (((output - target) ** 2) * mask_b1hw).sum() / valid

def masked_mae_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    data: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Pixelwise MAE reduced by mask (mean over valid points).
    """
    output, target = _to_b1hw(output), _to_b1hw(target)
    mask = _extract_mask(mask, data)
    mask_b1hw, valid = _valid_points_from_mask(mask, output, eps=eps)
    return (torch.abs(output - target) * mask_b1hw).sum() / valid


def sobel_gradient_magnitude_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    data: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    normalize: bool = False,
    regularization: str = "l2",
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Gradient-magnitude loss using Sobel filters.
    Matches the previous in-trainer `gradient_loss` (including optional normalization).
    """
    output, target = _to_b1hw(output), _to_b1hw(target)
    mask = _extract_mask(mask, data)
    mask_b1hw, valid = _valid_points_from_mask(mask, output, eps=eps)

    kx = _SOBEL_X.to(device=output.device, dtype=output.dtype)
    ky = _SOBEL_Y.to(device=output.device, dtype=output.dtype)
    grad_output_x = F.conv2d(output, kx, padding=1)
    grad_output_y = F.conv2d(output, ky, padding=1)
    grad_target_x = F.conv2d(target, kx, padding=1)
    grad_target_y = F.conv2d(target, ky, padding=1)
    
    # L1 norm
    if regularization == "l1":
        grad_x = torch.abs(grad_output_x - grad_target_x)
        grad_y = torch.abs(grad_output_y - grad_target_y)
        total_grad = grad_x + grad_y
    # L2 norm
    elif regularization == "l2":
        grad_x = (grad_output_x - grad_target_x)**2
        grad_y = (grad_output_y - grad_target_y)**2
        total_grad = grad_x + grad_y
    #L infinity norm
    elif regularization == "linf":
        grad_x = torch.abs(grad_output_x - grad_target_x)
        grad_y = torch.abs(grad_output_y - grad_target_y)
        total_grad = torch.max(grad_x, grad_y)
    elif regularization == "magnitude":
        mag_output = torch.sqrt(grad_output_x**2 + grad_output_y**2)
        mag_target = torch.sqrt(grad_target_x**2 + grad_target_y**2)
        total_grad = (mag_output - mag_target)**2
    else:
        raise ValueError(f"Invalid regularization type: {regularization!r}. Expected 'l1', 'l2', 'linf', or 'magnitude'")

    if normalize:
        valid_vals = total_grad[mask_b1hw.bool()]
        mean = valid_vals.mean()
        std = valid_vals.std()
        total_grad = (total_grad - mean) / (std + eps)

    return (total_grad * mask_b1hw).sum() / valid / _VAR_GRAD


def laplacian_curvature_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    data: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    normalize: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Curvature loss using Laplacian filter.
    Matches the previous in-trainer `curvature_loss` (including normalization).
    """
    output, target = _to_b1hw(output), _to_b1hw(target)
    mask = _extract_mask(mask, data)
    mask_b1hw, valid = _valid_points_from_mask(mask, output, eps=eps)

    k = _LAPLACE.to(device=output.device, dtype=output.dtype)
    curv_output = F.conv2d(output, k, padding=1)
    curv_target = F.conv2d(target, k, padding=1)

    if normalize:
        valid_vals = curv_output[mask_b1hw.bool()]
        valid_targ = curv_target[mask_b1hw.bool()]
        mean_output = valid_vals.mean()
        mean_target = valid_targ.mean()
        std_output = valid_vals.std()
        std_target = valid_targ.std()
        curv_output = (curv_output - mean_output) / (std_output + eps)
        curv_target = (curv_target - mean_target) / (std_target + eps)

    return (((curv_output - curv_target) ** 2) * mask_b1hw).sum() / valid / _VAR_CURV

def spectral_logloss(
    output: torch.Tensor,
    target: torch.Tensor,
    data: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Spectral log loss.
    """
    output, target = _to_b1hw(output), _to_b1hw(target)
    mask = _extract_mask(mask, data)
    mask_b1hw, valid = _valid_points_from_mask(mask, output, eps=eps)
    
    # 0. Apply Hanning window to prevent edge effects
    H, W = output.shape[-2], output.shape[-1]
    window_h = torch.hann_window(H, device=output.device).view(-1, 1)
    window_w = torch.hann_window(W, device=output.device).view(1, -1)
    window = window_h * window_w

    # 1. Compute the 2D Real FFT
    freq_output = torch.fft.rfft2(output * mask_b1hw * window, norm='ortho')
    freq_target = torch.fft.rfft2(target * mask_b1hw * window, norm='ortho')

    # 2. Compute Magnitude Spectrum
    mag_output = torch.abs(freq_output)**2
    mag_target = torch.abs(freq_target)**2

    # 3. Apply Log transform
    log_mag_output = torch.log(mag_output + eps)
    log_mag_target = torch.log(mag_target + eps)

    # 4. Compute the loss
    return ((log_mag_output - log_mag_target) ** 2).mean()

def spectral_slope_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    data: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-10,
    k_min: float = 0.05,
    k_max: float = 0.2
) -> torch.Tensor:
    """
    Spectral slope loss.
    """
    output, target = _to_b1hw(output), _to_b1hw(target)
    mask = _extract_mask(mask, data)
    mask_b1hw, valid = _valid_points_from_mask(mask, output, eps=eps)
    
    # 0. Apply Hanning window to prevent edge effects
    H, W = output.shape[-2], output.shape[-1]
    window_h = torch.hann_window(H, device=output.device).view(-1, 1)
    window_w = torch.hann_window(W, device=output.device).view(1, -1)
    window = window_h * window_w

    freq_output = torch.fft.rfft2(output * mask_b1hw * window, norm='ortho')
    freq_target = torch.fft.rfft2(target * mask_b1hw * window, norm='ortho')

    log_power_output = torch.log(torch.abs(freq_output)**2 + eps)
    log_power_target = torch.log(torch.abs(freq_target)**2 + eps)

    kx = torch.fft.rfftfreq(W, device=output.device)  # shape (W//2+1,)
    ky = torch.fft.fftfreq(H, device=output.device)   # shape (H,)
    KX, KY = torch.meshgrid(ky, kx, indexing='ij')  # shape (H, W//2+1)
    k_mag = torch.sqrt(KX**2 + KY**2).clamp(min=1e-10)  # avoid log(0) at DC

    log_k = torch.log(k_mag)  # shape (H, W//2+1)

    # Fit linear slope in log-log space within submesoscale band only
    d_log_power_output = log_power_output[..., 1:] - log_power_output[..., :-1]
    d_log_power_target = log_power_target[..., 1:] - log_power_target[..., :-1]
    d_log_k = (log_k[..., 1:] - log_k[..., :-1]).clamp(min=1e-10)

    slope_output = d_log_power_output / d_log_k
    slope_target = d_log_power_target / d_log_k

    diff = (slope_output - slope_target)**2

    return diff.mean() / (slope_target.var() + eps)


# ----------------------------
# Composition / weighted combo
# ----------------------------

@dataclass(frozen=True)
class WeightedLossTerm:
    name: str
    weight: float = 1.0
    args: Optional[Mapping[str, Any]] = None


class WeightedSumLoss(torch.nn.Module):
    """
    Weighted sum of multiple loss terms.

    After each forward call, stores:
    - `last_components`: dict[name] -> {"raw": Tensor, "weighted": Tensor, "contribution": Tensor}
        where:
          raw = loss_i(...)
          weighted = weight_i * raw
          contribution = weighted / (sum_w + eps) if normalize else weighted
    """

    def __init__(
        self,
        terms: Sequence[Union[Mapping[str, Any], WeightedLossTerm]],
        normalize: bool = True,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.normalize = bool(normalize)
        self.eps = float(eps)

        parsed_terms: List[WeightedLossTerm] = []
        for t in terms:
            if isinstance(t, WeightedLossTerm):
                parsed_terms.append(t)
                continue
            if not isinstance(t, dict):
                raise TypeError(f"Each loss term must be a dict, got {type(t)}")
            name = t.get("name") or t.get("type")
            if not isinstance(name, str):
                raise ValueError(f"Loss term missing 'name': {t}")
            weight = float(t.get("weight", 1.0))
            args = t.get("args") or {}
            parsed_terms.append(WeightedLossTerm(name=name, weight=weight, args=args))

        self._terms: List[Tuple[str, float, Callable[..., torch.Tensor]]] = [
            (term.name, term.weight, _resolve_loss_callable(term.name, term.args)) for term in parsed_terms
        ]

        self.last_components: Dict[str, Dict[str, torch.Tensor]] = {}

    def forward(self, output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        total = None
        wsum = 0.0
        comps: Dict[str, Dict[str, torch.Tensor]] = {}

        for name, w, fn in self._terms:
            if w == 0.0:
                continue
            raw = fn(output, target, **kwargs)
            weighted = raw * w
            comps[name] = {"raw": raw.detach(), "weighted": weighted.detach()}
            total = weighted if total is None else (total + weighted)
            wsum += w

        if total is None:
            raise ValueError("Weighted loss has no non-zero terms")

        if self.normalize:
            denom = total.new_tensor(wsum + self.eps)
            out = total / denom
            for v in comps.values():
                v["contribution"] = v["weighted"] / denom
        else:
            out = total
            for v in comps.values():
                v["contribution"] = v["weighted"]

        self.last_components = comps
        return out


def weighted_sum_loss_factory(
    terms: Sequence[Union[Mapping[str, Any], WeightedLossTerm]],
    normalize: bool = True,
    eps: float = 1e-10,
) -> Callable[..., torch.Tensor]:
    """
    Build a weighted loss:
        sum_i w_i * loss_i(...) / (sum_i w_i + eps)   if normalize=True
        sum_i w_i * loss_i(...)                      if normalize=False
    """
    # Backwards-compatible functional wrapper.
    module = WeightedSumLoss(terms, normalize=normalize, eps=eps)

    def _loss(output: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return module(output, target, **kwargs)

    # expose breakdown on the returned callable
    _loss._weighted_sum_module = module  # type: ignore[attr-defined]
    return _loss


