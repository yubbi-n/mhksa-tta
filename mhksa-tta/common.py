"""공통 유틸: config 로드, 모델/데이터셋 생성, device."""
from typing import Any
import copy
import json

import torch
import yaml

from dataset import get_datasets
from model import create_regressor
from .tap import MHKSAModel
from .encoder import MHKSAConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) if path.endswith(".json") else yaml.safe_load(f)


def build_model(config: dict[str, Any], load_adapter: bool) -> MHKSAModel:
    regressor = create_regressor(config)
    regressor.load_state_dict(torch.load(p := config["regressor"]["source"],
                                         map_location="cpu"))
    print(f"load regressor: {p}")

    mc = config["mhksa"]
    model = MHKSAModel(regressor,
                       backbone=config["regressor"]["config"].get("backbone"),
                       layer=mc.get("layer"),
                       n_cells=mc["n_cells"],
                       beta_init=mc["adapter"].get("beta_init", 0.0),
                       alpha_init=mc["adapter"].get("alpha_init", 4.0))
    if load_adapter:
        sd = torch.load(p := mc["adapter_ckpt"], map_location="cpu")
        model.lagm.load_state_dict(sd)
        print(f"load LAGM adapter: {p}  "
              f"(beta_LAP={model.lagm.beta_lap.item():.4f}, "
              f"alpha mean={model.lagm.alpha.mean().item():.4f})")
    return model.to(DEVICE)


def encoder_config(config: dict[str, Any], cosine: bool | None = None) -> MHKSAConfig:
    ec = dict(config["mhksa"]["encoder"])
    if "m_range" in ec:
        ec["m_range"] = tuple(ec["m_range"])
    if cosine is not None:
        ec["cosine"] = cosine
    return MHKSAConfig(**ec)


def source_dataset(config: dict[str, Any], train_aug: bool = False):
    """source train split (stats 계산은 augmentation 없이)."""
    c = copy.deepcopy(config)
    c["dataset"] = copy.deepcopy(config["source_dataset"])
    c["dataset"]["train_aug"] = train_aug
    return get_datasets(c)[0]


def target_dataset(config: dict[str, Any]):
    """SSA adaptation.py와 동일하게 target의 val split 사용."""
    c = copy.deepcopy(config)
    c["dataset"] = copy.deepcopy(config["target_dataset"])
    return get_datasets(c)[1]


def freeze(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
