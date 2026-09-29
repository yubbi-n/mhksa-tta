"""§5-8  Wiener filter → multi-head RFF → per-head PCA → concat → final PCA.

g: [B, D] (LAGM 출력) → Ψ(g): [B, K]   (전부 torch 연산이라 TTA 때 backprop 가능)

구현 메모
- Wiener 흡수(§6):  W_h^new = W_h U_g diag(w) U_g^T  를 [M×D] 행렬로 만들지 않고
  t = U_g^T g  (K_W차원) 공간에서 계산 →  W_h^new g = (W_h U_g) (w ⊙ t).
  U_g 열이 orthonormal이므로 수학적으로 문서와 동일하고, D=16384 같은 경우에도 메모리 절약.
- cosine=True (MHKSA-cosine): RFF 입력을 g_wiener / ||g_wiener|| 로 L2 정규화
  (원논문 CoRP = φ_RFF(φ_cos(z)) 대응).  ||g_wiener|| = ||w ⊙ t||.
  cosine=False (MHKSA-plain): 문서 그대로.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn, Tensor


@dataclass
class MHKSAConfig:
    n_heads: int = 4            # H
    rff_dim: int = 256          # M
    wiener_rank: int = 1000     # K_W
    head_rank: int = 128        # P
    final_rank: int = 100       # K
    eta: float = 1.0            # Wiener shrinkage exponent
    m_range: tuple[float, float] = (0.3, 3.0)   # m_h ∈ [0.3, 3.0] log-uniform
    bandwidth_mode: str = "random"   # "random"(log-uniform 샘플) | "logspace"(등간격)
    cosine: bool = False        # MHKSA-cosine vs MHKSA-plain
    ridge_tau: float = 1e-4     # τ_R
    max_pca_samples: int = 4096     # Wiener Gram trick subsample (MAX_PCA_SAMPLES)
    max_median_samples: int = 2048  # median heuristic용 subsample
    noise_decile: float = 0.1   # σ² = bottom-decile mean


class MHKSAEncoder(nn.Module):
    """학습 파라미터 없음. 모든 텐서는 buffer (source stats, frozen)."""

    def __init__(self, cfg: MHKSAConfig):
        super().__init__()
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # encoding
    # ------------------------------------------------------------------ #
    def wiener_coords(self, g: Tensor) -> Tensor:
        """g [B, D] → w ⊙ (U_g^T g)  [B, K_W]  (g_wiener의 U_g 좌표)."""
        return (g @ self.U_g) * self.w

    def rff(self, g: Tensor) -> Tensor:
        """g [B, D] → φ_h(g_wiener)  [B, H, M]   (§5 + §6 흡수)."""
        tw = self.wiener_coords(g)
        if self.cfg.cosine:
            tw = tw / (tw.norm(dim=1, keepdim=True) + 1e-10)
        proj = torch.einsum("bk,hmk->bhm", tw, self.WU)        # W_h g_wiener
        M = self.WU.shape[1]
        return math.sqrt(2.0 / M) * torch.cos(proj + self.b)

    def head_pca(self, phi: Tensor) -> Tensor:
        """φ [B, H, M] → u(g) [B, H*P]   (§7)."""
        phi_c = phi - self.mu_phi                               # centering
        u = torch.einsum("bhm,hmp->bhp", phi_c, self.B_h)
        return u.flatten(1)

    def forward(self, g: Tensor) -> Tensor:
        """g [B, D] → Ψ(g) [B, K]   (§8)."""
        return self.head_pca(self.rff(g)) @ self.B_final

    # ------------------------------------------------------------------ #
    # fitting (Stage 2, source stats)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def fit_wiener_and_bandwidth(self, g_sub: Tensor, seed: int = 0):
        """g_sub: [n, D] source subsample (n ≤ max_pca_samples)."""
        cfg = self.cfg
        dev = g_sub.device
        g64 = g_sub.double()
        n, D = g64.shape

        # ---- §6 Step 1-4: Gram trick
        Xc = g64 - g64.mean(dim=0, keepdim=True)                 # X̃ [n, D]
        Q = Xc @ Xc.T                                            # [n, n]
        q, U_Q = torch.linalg.eigh(Q)                            # 오름차순
        K_W = min(cfg.wiener_rank, n - 1)
        q, U_Q = q.flip(0)[:K_W].clamp_min(1e-12), U_Q.flip(1)[:, :K_W]
        U_g = Xc.T @ U_Q / q.sqrt()                              # [D, K_W] orthonormal
        lam = q / (n - 1)                                        # unbiased (§0)

        # noise floor: bottom-decile mean
        n_tail = max(1, int(round(cfg.noise_decile * K_W)))
        sigma2 = max(lam[-n_tail:].mean().item(), 1e-8)
        w = (lam / (lam + sigma2)).pow(cfg.eta)                  # [K_W]

        self.register_buffer("U_g", U_g.float())
        self.register_buffer("w", w.float())
        self.register_buffer("wiener_lam", lam.float())
        self.wiener_sigma2 = sigma2

        # ---- §5 bandwidth: γ_med = 1 / (2 d_med²)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        m = min(cfg.max_median_samples, n)
        idx = torch.randperm(n, generator=gen)[:m].to(dev)
        if cfg.cosine:   # RFF 입력 공간(정규화된 g_wiener)에서 거리 계산
            x = self.wiener_coords(g_sub[idx].float()).double()
            x = x / (x.norm(dim=1, keepdim=True) + 1e-10)
        else:            # 문서: 원본 g의 pairwise distance
            x = g64[idx]
        dist = torch.cdist(x, x)
        iu = torch.triu_indices(m, m, offset=1, device=dev)
        d_med = dist[iu[0], iu[1]].median().item()
        gamma_med = 1.0 / (2.0 * d_med ** 2)

        lo, hi = cfg.m_range
        if cfg.bandwidth_mode == "logspace":
            mult = torch.logspace(math.log10(lo), math.log10(hi), cfg.n_heads,
                                  dtype=torch.float64)
        else:
            mult = torch.exp(torch.empty(cfg.n_heads, dtype=torch.float64)
                             .uniform_(math.log(lo), math.log(hi), generator=gen))
        gammas = gamma_med * mult                                # γ_h

        # ---- §5 random weights: W_h ~ N(0, 2γ_h I), b_h ~ U(0, 2π)
        H, M = cfg.n_heads, cfg.rff_dim
        WU = torch.empty(H, M, K_W, dtype=torch.float64)
        for h in range(H):
            W_h = torch.randn(M, D, generator=gen, dtype=torch.float64) \
                * math.sqrt(2.0 * gammas[h].item())
            WU[h] = W_h @ U_g.cpu()                               # W_h U_g [M, K_W]
        b = 2 * math.pi * torch.rand(H, M, generator=gen, dtype=torch.float64)

        self.register_buffer("WU", WU.float().to(dev))
        self.register_buffer("b", b.float().to(dev))
        self.register_buffer("gammas", gammas.float())
        self.d_med, self.gamma_med = d_med, gamma_med

    @torch.no_grad()
    def fit_head_pca(self, phi: Tensor):
        """phi: [N_s, H, M] (전체 source의 RFF feature, CPU OK)."""
        N_s, H, M = phi.shape
        P = min(self.cfg.head_rank, M)
        mu = phi.double().mean(dim=0)                            # μ_h^φ [H, M]
        B_h = torch.empty(H, M, P, dtype=torch.float64)
        lam_h = torch.empty(H, P, dtype=torch.float64)
        for h in range(H):
            pc = phi[:, h].double() - mu[h]
            cov = pc.T @ pc / (N_s - 1)                          # Σ_h^φ [M, M]
            ev, U = torch.linalg.eigh(cov)
            B_h[h] = U.flip(1)[:, :P]
            lam_h[h] = ev.flip(0)[:P]
        dev = self.U_g.device
        self.register_buffer("mu_phi", mu.float().to(dev))
        self.register_buffer("B_h", B_h.float().to(dev))
        self.register_buffer("lam_h", lam_h.float().to(dev))

    @torch.no_grad()
    def fit_final_pca(self, u: Tensor) -> Tensor:
        """u: [N_s, HP] → Ψ_S [N_s, K]  (u는 이미 source-centered)."""
        N_s, HP = u.shape
        K = min(self.cfg.final_rank, HP)
        if K < self.cfg.final_rank:
            print(f"[MHKSA] final_rank {self.cfg.final_rank} > H*P={HP} → K={K}")
        u64 = u.double()
        cov = u64.T @ u64 / (N_s - 1)                            # Σ^ph [HP, HP]
        ev, V = torch.linalg.eigh(cov)
        B_final = V.flip(1)[:, :K]
        lam_ph = ev.flip(0)[:K]
        dev = self.U_g.device
        self.register_buffer("B_final", B_final.float().to(dev))
        self.register_buffer("lam_ph", lam_ph.float().to(dev))
        return (u64 @ B_final).float()

    @torch.no_grad()
    def fit_ridge_probe(self, psi: Tensor, y: Tensor):
        """§10 c_probe = (Ψ^TΨ + τ_R I)^{-1} Ψ^T y,   ω_k = 1 + |c_probe[k]|."""
        psi64, y64 = psi.double(), y.double().flatten()
        y64 = y64 - y64.mean()        # Ψ가 zero-mean이므로 c에는 영향 없음 (intercept 흡수)
        K = psi64.shape[1]
        A = psi64.T @ psi64 + self.cfg.ridge_tau * torch.eye(K, dtype=torch.float64)
        c = torch.linalg.solve(A, psi64.T @ y64)
        dev = self.U_g.device
        self.register_buffer("c_probe", c.float().to(dev))
        self.register_buffer("omega", (1.0 + c.abs()).float().to(dev))

    # ------------------------------------------------------------------ #
    # save / load
    # ------------------------------------------------------------------ #
    BUFFERS = ["U_g", "w", "wiener_lam", "WU", "b", "gammas", "mu_phi", "B_h",
               "lam_h", "B_final", "lam_ph", "c_probe", "omega"]

    def export(self) -> dict:
        return {
            "cfg": vars(self.cfg),
            "buffers": {k: getattr(self, k).cpu() for k in self.BUFFERS},
            "scalars": {"wiener_sigma2": self.wiener_sigma2,
                        "d_med": self.d_med, "gamma_med": self.gamma_med},
        }

    @classmethod
    def load(cls, state: dict) -> "MHKSAEncoder":
        cfg = dict(state["cfg"])
        cfg["m_range"] = tuple(cfg["m_range"])
        enc = cls(MHKSAConfig(**cfg))
        for k, v in state["buffers"].items():
            enc.register_buffer(k, v)
        for k, v in state["scalars"].items():
            setattr(enc, k, v)
        return enc
