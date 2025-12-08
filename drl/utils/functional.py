from typing import Tuple, Union, Optional

import torch


def gaussian_kl_divergence(
    mu1: torch.Tensor, 
    std1: torch.Tensor, 
    mu2: torch.Tensor, 
    std2: torch.Tensor,
    *,
    dim: int = -1,
    keepdim: bool = True
) -> torch.Tensor:
    """ Compute KL Divergence between two Gaussian distributions. """
    return torch.sum(
        (std2 / std1).log() + (std1.pow(2) + (mu1 - mu2).pow(2)) / (2 * std2.pow(2)) - 0.5, 
        dim=dim, keepdim=keepdim
    )


def mean_square_error(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    *,
    dim: int = -1, 
    keepdim: bool = False
) -> torch.Tensor:
    """ Compute Measure Square Error betwee predictions and targets. """
    return torch.sum((preds - targets).square(), dim=dim, keepdim=keepdim)


def smooth_l1(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    *,
    beta: Union[torch.Tensor, float] = 1.0,
    dim: int = -1, 
    keepdim: bool = False
) -> torch.Tensor:
    """ Compute Smooth L1 Loss between predictions and targets. """
    diff = preds - targets
    abs_diff = diff.abs()
    loss = torch.where(abs_diff < beta, 0.5 * diff.square() / beta, abs_diff - 0.5 * beta)
    return torch.sum(loss, dim=dim, keepdim=keepdim)


def clip_coef(
    coefs: torch.Tensor, 
    deltas: torch.Tensor, 
    *,
    coef_range: Tuple[float, float]
) -> torch.Tensor:
    """ Compute clipping error values. """
    unclip_losses = coefs * deltas
    clip_lossses = coefs.clamp(*coef_range) * deltas
    return -torch.minimum(unclip_losses, clip_lossses)


def symmetric_info_nce(
    logits: torch.Tensor, 
    *,
    penalty_w: float = 0.01
) -> torch.Tensor:
    """ Symmetric Info NCE for Contrastive Learning. """
    s = torch.diag(logits)
    s0 = torch.logsumexp(logits, dim=0)
    s1 = torch.logsumexp(logits, dim=1)
    return ((s0 + s1 - 2 * s) + penalty_w * (s0**2 + s1**2)) / 2


def reduce_by_mean(
    losses: torch.Tensor,
    *,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """ Reduce losses by mean, optionally weighted. """
    weights = torch.ones_like(losses) if weights is None else weights.view(losses.size())
    return torch.mean(losses * weights)


def group_by_mean(
    losses: torch.Tensor,
    group_idxs: torch.Tensor,
    *,
    weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """ Reduce losses by mean within groups defined by group_idxs, optionally weighted. """
    weights = torch.ones_like(losses) if weights is None else weights.view(losses.size())
    uniques, inverse_idxs, counts = torch.unique(group_idxs, return_inverse=True, return_counts=True)
    M = torch.zeros(len(uniques), len(losses), dtype=losses.dtype, device=losses.device)
    M[inverse_idxs, torch.arange(len(losses))] = weights.view(-1)
    return torch.sum(M * losses.view(-1), dim=-1) / counts