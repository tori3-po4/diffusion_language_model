"""Log-linear noise schedule and forward diffusion for MDLM."""

import torch
import torch.nn.functional as F


class LogLinearNoise:
    """Log-linear noise schedule.

    sigma(t) = -log(1 - (1 - eps) * t)
    move_chance(t) = 1 - exp(-sigma(t)) = (1 - eps) * t

    This gives a linear masking rate in t, simplifying the sampler.
    """

    def __init__(self, eps: float = 1e-3):
        self.eps = eps

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sigma(t) = -log(1 - (1-eps)*t)."""
        return -torch.log1p(-(1 - self.eps) * t)

    def move_chance(self, t: torch.Tensor) -> torch.Tensor:
        """Probability that each token is masked at time t."""
        return (1 - self.eps) * t

    def rate(self, t: torch.Tensor) -> torch.Tensor:
        """dsigma/dt = (1 - eps) / (1 - (1 - eps) * t)."""
        return (1 - self.eps) / (1 - (1 - self.eps) * t)


def q_xt(x0: torch.Tensor, move_chance: torch.Tensor, mask_token_id: int) -> torch.Tensor:
    """Forward diffusion: independently mask each token with probability move_chance.

    Args:
        x0: Clean token indices, shape (B, S).
        move_chance: Per-sample masking probability, shape (B, 1) or (B, S).
        mask_token_id: Token ID used for [MASK].

    Returns:
        xt: Noisy token indices, shape (B, S).
    """
    mask = torch.rand_like(x0.float()) < move_chance
    xt = torch.where(mask, mask_token_id, x0)
    return xt
