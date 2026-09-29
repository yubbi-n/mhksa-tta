"""Stage 1 — LAGM adapter pretrain (β_LAP, α̃ 학습).

source model(backbone+regressor)은 완전히 frozen(eval 모드).
LAGM 출력 g 위에 임시 linear head를 붙여 source label로 MSE 학습 → β_LAP, α̃만 저장.
(head는 버림. 계획서에 Stage 1 objective가 명시돼 있지 않아 source-label MSE로 가정)

python3 mhksa_train_adapter.py -c configs/mhksa/svhn.yaml -o result/mhksa/svhn
"""
from pathlib import Path
from pprint import pprint
import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from utils.seed import fix_seed
from mhksa.common import DEVICE, load_config, build_model, source_dataset, freeze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", required=True)
    ap.add_argument("-o", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha_init", type=float, default=None,
                    help="config의 alpha_init override (4→α≈0.982, 0→0.5)")
    args = ap.parse_args()
    pprint(vars(args))
    fix_seed(args.seed)

    config = load_config(args.c)
    ac = config["mhksa"]["adapter"]
    if args.alpha_init is not None:
        ac["alpha_init"] = args.alpha_init
    Path(args.o).mkdir(parents=True, exist_ok=True)

    model = build_model(config, load_adapter=False)
    freeze(model.regressor)
    model.regressor.eval()

    ds = source_dataset(config, train_aug=ac.get("train_aug", False))
    dl = DataLoader(ds, batch_size=ac.get("batch_size", 256), shuffle=True,
                    num_workers=ac.get("num_workers", 4), drop_last=True)

    # D 확인용 dummy forward
    with torch.no_grad():
        x0, _ = next(iter(dl))
        D = model.lagm_only(x0[:2].to(DEVICE)).shape[1]
    print(f"LAGM output dim D = {D}")

    head = nn.Linear(D, 1).to(DEVICE)
    opt = torch.optim.Adam([
        {"params": model.lagm.parameters(), "lr": ac.get("lr", 1e-2)},
        {"params": head.parameters(), "lr": ac.get("head_lr", 1e-3)},
    ])

    for ep in range(ac.get("epochs", 5)):
        tot, n = 0.0, 0
        for it, (x, y) in enumerate(dl):
            x, y = x.to(DEVICE), y.float().flatten().to(DEVICE)
            g = model.lagm_only(x)
            loss = F.mse_loss(head(g).flatten(), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
            if ac.get("max_iters_per_epoch") and it + 1 >= ac["max_iters_per_epoch"]:
                break
        print(f"[epoch {ep}] mse={tot / n:.5f}  "
              f"beta_LAP={model.lagm.beta_lap.item():.4f}  "
              f"alpha(mean/min/max)={model.lagm.alpha.mean().item():.4f}/{model.lagm.alpha.min().item():.4f}/{model.lagm.alpha.max().item():.4f}",
              flush=True)

    p = Path(args.o, "lagm_adapter.pt")
    torch.save(model.lagm.state_dict(), p)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
