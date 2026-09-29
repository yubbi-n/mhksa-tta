"""§1-4  LAP (Local-Attention Pooling) + GAP + α gate → LAGM.

NasSA_Method v4 표기를 그대로 따름.
    Z      : [B, C, S_h, S_w]   backbone layer L 출력
    LAP    : [B, C, N, N]
    z_bar  : [B, C]              (GAP, backbone-mean)
    alpha  : [N, N] = sigmoid(alpha_tilde)
    G      : [B, C, N, N] = alpha * LAP + (1 - alpha) * z_bar
    g      : [B, D],  D = C * N^2
학습 파라미터: beta_lap (scalar), alpha_tilde [N, N]  (Stage 1에서 학습, 이후 frozen)
"""
import math

import torch
from torch import nn, Tensor


def adaptive_cell_bounds(size: int, n: int) -> list[tuple[int, int]]:
    """r_i^start = floor(i * S / N), r_i^end = ceil((i + 1) * S / N)."""
    return [(math.floor(i * size / n), math.ceil((i + 1) * size / n))
            for i in range(n)]


class LAGM(nn.Module):
    def __init__(self, n_cells: int, beta_init: float = 0.0,
                 alpha_init: float = 4.0):
        super().__init__()
        self.n = n_cells
        # β_LAP: 스칼라 1개 (0 → uniform, >0 → max-like, <0 → min-like)
        self.beta_lap = nn.Parameter(torch.tensor(float(beta_init)))
        # α̃: [N, N]  (--alpha_init; 4 → α≈0.982 거의 pure LAP, 0 → 0.5)
        self.alpha_tilde = nn.Parameter(
            torch.full((n_cells, n_cells), float(alpha_init)))

    @property
    def alpha(self) -> Tensor:
        return torch.sigmoid(self.alpha_tilde)

    def lap(self, z: Tensor) -> Tensor:
        """§1  z: [B, C, S_h, S_w] → LAP: [B, C, N, N]."""
        B, C, S_h, S_w = z.shape
        rows = adaptive_cell_bounds(S_h, self.n)
        cols = adaptive_cell_bounds(S_w, self.n)
        out = z.new_empty(B, C, self.n, self.n)
        for i, (r0, r1) in enumerate(rows):
            for j, (c0, c1) in enumerate(cols):
                cell = z[:, :, r0:r1, c0:c1]                  # [B, C, c_h, c_w]
                s_ij = cell.mean(dim=1)                       # [B, c_h, c_w] 채널평균
                w_ij = torch.softmax(
                    (self.beta_lap * s_ij).flatten(1), dim=1)  # [B, c_h*c_w]
                out[:, :, i, j] = torch.einsum(
                    "bcp,bp->bc", cell.flatten(2), w_ij)
        return out

    def forward(self, z: Tensor) -> Tensor:
        """z: [B, C, S_h, S_w] (또는 [B, C] tabular) → g: [B, D]."""
        if z.ndim == 2:                       # MLP feature → [B, C, 1, 1]
            z = z[:, :, None, None]
        lap = self.lap(z)                                     # [B, C, N, N]
        z_bar = z.mean(dim=(2, 3))                            # [B, C]   §2 GAP
        a = self.alpha                                        # [N, N]   §3
        G = a * lap + (1.0 - a) * z_bar[:, :, None, None]     # [B, C, N, N] §4
        return G.flatten(1)                                   # [B, C*N^2]
