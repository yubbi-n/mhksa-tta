"""Stage 3 — MHKSA TTA (§9-10).  SSA adaptation.py와 동일한 프로토콜:
target val split으로 1 epoch TTA(online metric) → 같은 데이터로 offline 평가.
업데이트 대상: feature extractor의 BN affine (γ_BN, β_BN).  LAGM/RFF/PCA는 frozen.

python3 mhksa_adapt.py -c configs/mhksa/svhn.yaml -o result/mhksa_tta/svhn_plain \
    --stats result/mhksa/svhn/mhksa_stats_plain.pt
"""
from pathlib import Path
from pprint import pprint
import argparse
import json

import yaml
import torch
from torch.utils.data import DataLoader
from ignite.engine import Engine
from ignite.metrics import RootMeanSquaredError, MeanAbsoluteError
from ignite.contrib.metrics.regression.r2_score import R2Score

from utils.seed import fix_seed
from adaptation import create_optimizer
from evaluation.evaluator import RegressionEvaluator
from mhksa.common import DEVICE, load_config, build_model, target_dataset, freeze
from mhksa.encoder import MHKSAEncoder
from mhksa.loss import mhksa_tta_loss


class MHKSATTAEngine(Engine):
    def __init__(self, model, encoder: MHKSAEncoder, opt, train_mode: bool = True,
                 eps: float = 1e-8, use_omega: bool = True):
        super().__init__(self.update)
        self.model, self.enc, self.opt = model, encoder, opt
        self.train_mode, self.eps = train_mode, eps
        self.omega = encoder.omega if use_omega else torch.ones_like(encoder.omega)
        y_ot = lambda d: (d["y_pred"], d["y"])
        RootMeanSquaredError(y_ot).attach(self, "rmse_loss")
        MeanAbsoluteError(y_ot).attach(self, "mae_loss")
        R2Score(y_ot).attach(self, "R2")

    def update(self, engine, batch):
        # regressor만 train 모드 (BN batch stats); LAGM/encoder는 파라미터 없음/frozen
        self.model.regressor.train(self.train_mode)
        self.opt.zero_grad()
        x, y = batch
        x = x.to(DEVICE)

        y_pred, _, g = self.model(x)
        psi = self.enc(g)                                     # [N_B, K]
        loss, ell = mhksa_tta_loss(psi, self.enc.lam_ph, self.omega, self.eps)
        loss.backward()
        self.opt.step()

        if engine.state.iteration % 50 == 1:
            print(f"[iter {engine.state.iteration}] loss={loss.item():.4f} "
                  f"ℓ[:5]={ell[:5].detach().cpu().numpy().round(3)}", flush=True)
        return {"y_pred": y_pred.detach(), "y": y.float().flatten().to(DEVICE)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", required=True)
    ap.add_argument("-o", required=True)
    ap.add_argument("--stats", required=True, help="mhksa_stats_{plain,cosine}.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_omega", action="store_true", help="ablation: ω_k = 1")
    args = ap.parse_args()
    pprint(vars(args))
    fix_seed(args.seed)

    config = load_config(args.c)
    Path(args.o).mkdir(parents=True, exist_ok=True)
    with Path(args.o, "config.yaml").open("w", encoding="utf-8") as f:
        yaml.dump({**config, "args": vars(args)}, f)

    model = build_model(config, load_adapter=True)
    freeze(model.lagm)
    enc = MHKSAEncoder.load(torch.load(args.stats, map_location="cpu")).to(DEVICE)
    print(f"encoder: cosine={enc.cfg.cosine}, K={enc.lam_ph.numel()}, "
          f"H={enc.WU.shape[0]}, M={enc.WU.shape[1]}, K_W={enc.U_g.shape[1]}")

    tc = config["mhksa"]["tta"]
    opt = create_optimizer(model.regressor, {"optimizer": tc["optimizer"]})

    val_ds = target_dataset(config)
    adapt_dl = DataLoader(val_ds, **tc["adapt_dataloader"])
    eval_dl = DataLoader(val_ds, **tc.get("val_dataloader", {"batch_size": 256}))

    engine = MHKSATTAEngine(model, enc, opt, train_mode=tc.get("train_mode", True),
                            eps=tc.get("eps", 1e-8), use_omega=not args.no_omega)
    evaluator = RegressionEvaluator(model.regressor)

    engine.run(adapt_dl)
    evaluator.run(eval_dl)

    metrics = {"iteration": engine.state.iteration,
               "online": engine.state.metrics,
               "offline": evaluator.state.metrics}
    pprint(metrics)
    with Path(args.o, "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
