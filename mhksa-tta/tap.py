"""Backbone layer L 출력 Z와 penultimate feature를 forward 한 번으로 같이 뽑는 wrapper.

regressor 예측(y_pred)은 원래 SSA와 동일하게 GAP penultimate feature → linear 경로를 쓰고,
LAGM 브랜치(g)는 TTA loss 계산에만 사용 (§10 주의: Ψ는 regressor path 밖).

SSA repo의 CNNRegressor.feature_extractor는 FX GraphModule이라 forward hook이 안 걸림 →
같은 GraphModule에서 {layer L 마지막 노드, 최종 노드}를 반환하는 extractor를 다시 만든다.
(submodule 객체를 공유하므로 BN affine 업데이트가 regressor에 그대로 반영됨)
"""
from torch import nn, Tensor
from torchvision.models.feature_extraction import (create_feature_extractor,
                                                   get_graph_node_names)

from model import Regressor, CNNRegressor, MLPRegressor
from .lagm import LAGM


class MHKSAModel(nn.Module):
    def __init__(self, regressor: Regressor, backbone: str | None,
                 layer: int | None, n_cells: int,
                 beta_init: float = 0.0, alpha_init: float = 4.0):
        super().__init__()
        self.regressor = regressor
        self.lagm = LAGM(n_cells, beta_init, alpha_init)

        if isinstance(regressor, CNNRegressor):
            fe = regressor.get_feature_extractor()
            names = get_graph_node_names(fe)[0]
            layer_nodes = [n for n in names if n.startswith(f"layer{layer}.")]
            assert layer_nodes, f"layer{layer} not found in {backbone}"
            self.z_node = layer_nodes[-1]
            self.tap = create_feature_extractor(
                fe, {self.z_node: "z", names[-1]: "feature"})
            assert next(self.tap.parameters()) is next(fe.parameters())
            self._is_mlp = False
            print(f"[MHKSA] tap layer L={layer}: node {self.z_node!r}")
        elif isinstance(regressor, MLPRegressor):
            # tabular: penultimate feature [B, C] 를 [B, C, 1, 1]로 취급
            self._is_mlp = True
        else:
            raise TypeError(type(regressor))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """returns (y_pred [B], penultimate feature [B, D_pen], g [B, D])."""
        if self._is_mlp:
            feat = self.regressor.feature(x)
            z_map = feat
        else:
            out = self.tap(x)
            feat, z_map = out["feature"].flatten(1), out["z"]
        y_pred = self.regressor.predict_from_feature(feat)
        return y_pred, feat, self.lagm(z_map)

    def lagm_only(self, x: Tensor) -> Tensor:
        return self.forward(x)[2]
