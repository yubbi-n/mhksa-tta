"""§9-10  Target batch stats + dim-weighted symmetric Gaussian KL."""
import torch
from torch import Tensor


def batch_stats(z: Tensor) -> tuple[Tensor, Tensor]:
    """§9  z̃^t [N_B, K] → μ̂^t, σ̂^{t,2}  (1/N_B, biased — 문서 §9 식 그대로)."""
    mu = z.mean(dim=0)
    var = (z - mu).square().mean(dim=0)
    return mu, var


def symmetric_kl(mu_t: Tensor, var_t: Tensor, lam_s: Tensor,
                 eps: float = 1e-8) -> Tensor:
    """§10  ℓ_k = ½ [ ((μ̂)² + λ_k^ph) / σ̂² + ((μ̂)² + σ̂²) / λ_k^ph − 2 ]   → [K]
    (source 쪽 mean=0, var=λ^ph;  KL(src‖tgt) + KL(tgt‖src), log 항은 상쇄)."""
    m2 = mu_t.square()
    return 0.5 * ((m2 + lam_s) / (var_t + eps) + (m2 + var_t) / (lam_s + eps) - 2.0)


def mhksa_tta_loss(psi_t: Tensor, lam_s: Tensor, omega: Tensor,
                   eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """ℒ_TTA = Σ_k ω_k ℓ_k.   returns (loss, per-dim ℓ)."""
    mu, var = batch_stats(psi_t)
    ell = symmetric_kl(mu, var, lam_s, eps)
    return ell @ omega, ell
