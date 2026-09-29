"""Stage 2 — source statistics (§5-8 fit, §10 ridge probe).

  1) source subsample(≤4096)의 g → Wiener (Gram trick) + median heuristic + RFF 샘플링
  2) 전체 source의 φ_h(g) 수집 → per-head PCA
  3) u(g) → final PCA → Ψ_S → ridge probe → ω

python3 mhksa_stats.py -c configs/mhksa/svhn.yaml -o result/mhksa/svhn            # plain
python3 mhksa_stats.py -c configs/mhksa/svhn.yaml -o result/mhksa/svhn --cosine   # cosine
"""
from pathlib import Path
from pprint import pprint
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from utils.seed import fix_seed
from mhksa.common import (DEVICE, load_config, build_model, source_dataset,
                          encoder_config, freeze)
from mhksa.encoder import MHKSAEncoder


@torch.no_grad()
def collect(model, dl, fn):
    outs, ys = [], []
    for x, y in dl:
        outs.append(fn(model.lagm_only(x.to(DEVICE))).cpu())
        ys.append(y.float().flatten())
    return torch.cat(outs), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", required=True)
    ap.add_argument("-o", required=True)
    ap.add_argument("--seed", type=int, default=0, help="RFF 샘플링 seed")
    ap.add_argument("--cosine", action="store_true", help="MHKSA-cosine 변형")
    args = ap.parse_args()
    pprint(vars(args))
    fix_seed(args.seed)

    config = load_config(args.c)
    Path(args.o).mkdir(parents=True, exist_ok=True)
    cfg = encoder_config(config, cosine=args.cosine or None)
    pprint(vars(cfg))

    model = build_model(config, load_adapter=True)
    freeze(model)
    model.eval()

    ds = source_dataset(config, train_aug=False)
    bs = config["mhksa"].get("stats_batch_size", 256)
    nw = config["mhksa"].get("num_workers", 4)
    rng = np.random.default_rng(args.seed)

    max_src = config["mhksa"].get("max_source_samples")
    if max_src and len(ds) > max_src:
        ds = Subset(ds, rng.choice(len(ds), max_src, replace=False).tolist())
    N_s = len(ds)
    print(f"N_s = {N_s}")

    # ---- 1) Wiener + bandwidth + RFF weights
    n_sub = min(cfg.max_pca_samples, N_s)
    sub = Subset(ds, rng.choice(N_s, n_sub, replace=False).tolist())
    g_sub, _ = collect(model, DataLoader(sub, batch_size=bs, num_workers=nw),
                       lambda g: g)
    print(f"g subsample: {tuple(g_sub.shape)}  (D = {g_sub.shape[1]})")

    enc = MHKSAEncoder(cfg)
    enc.fit_wiener_and_bandwidth(g_sub.to(DEVICE), seed=args.seed)
    del g_sub
    print(f"Wiener: K_W={enc.U_g.shape[1]}, sigma2={enc.wiener_sigma2:.3e}, "
          f"w∈[{enc.w.min():.3f}, {enc.w.max():.3f}], "
          f"#(w>0.5)={(enc.w > 0.5).sum().item()}")
    print(f"median heuristic: d_med={enc.d_med:.4f}, gamma_med={enc.gamma_med:.3e}, "
          f"gamma_h={enc.gammas.numpy().round(6).tolist()}")

    # ---- 2) per-head PCA
    dl = DataLoader(ds, batch_size=bs, num_workers=nw)
    phi, y = collect(model, dl, enc.rff)                     # [N_s, H, M]
    enc.fit_head_pca(phi)
    print(f"per-head PCA: B_h {tuple(enc.B_h.shape)}; top-3 λ_h per head = "
          f"{enc.lam_h[:, :3].cpu().numpy().round(5).tolist()}")

    # ---- 3) final PCA + ridge probe
    u = torch.cat([enc.head_pca(p.to(DEVICE)).cpu() for p in phi.split(4096)])
    del phi
    psi = enc.fit_final_pca(u)                               # [N_s, K]
    enc.fit_ridge_probe(psi, y)
    print(f"final PCA: K={enc.lam_ph.numel()}, λ^ph[:5]={enc.lam_ph[:5].cpu().numpy().round(5)}, "
          f"λ^ph[-1]={enc.lam_ph[-1].item():.3e}")
    print(f"ridge probe: ω ∈ [{enc.omega.min():.3f}, {enc.omega.max():.3f}]")

    # ridge probe 적합도 (source R²) — 참고용
    pred = psi.double() @ enc.c_probe.cpu().double() + y.double().mean()
    r2 = 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"ridge probe source R² = {r2.item():.4f}")

    name = "mhksa_stats_cosine.pt" if cfg.cosine else "mhksa_stats_plain.pt"
    torch.save(enc.export(), p := Path(args.o, name))
    print(f"saved {p}")


if __name__ == "__main__":
    main()
