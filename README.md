# MHKSA-TTA (NasSA_Method v4 구현)

SSA 공식 코드(https://github.com/kzkadc/regression-tta) 위에 얹는 add-on.
repo를 clone한 뒤 이 폴더 내용을 **repo 루트에 그대로 복사**하면 됨 (기존 파일은 수정하지 않음).

```bash
git clone https://github.com/kzkadc/regression-tta && cd regression-tta
cp -r /path/to/mhksa-tta/* .
# dataset/dataset_config.py에 데이터 경로 설정 (SVHN source model은 result/source/svhn에 포함돼 있음)
bash run_mhksa.sh svhn
```

## 파일 ↔ 계획서 섹션
| 파일 | 내용 |
|---|---|
| `mhksa/lagm.py` | §1 LAP (adaptive cell, softmax β_LAP) · §2 GAP · §3 α gate · §4 LAGM → g |
| `mhksa/tap.py` | backbone layer L 출력 Z + penultimate feature를 한 번의 forward로 추출 |
| `mhksa/encoder.py` | §5 median heuristic / multi-head RFF · §6 Wiener (Gram trick, unbiased) · §7 per-head PCA · §8 final PCA · §10 ridge probe(ω) |
| `mhksa/loss.py` | §9 batch stats (1/N_B) · §10 symmetric KL, ℒ = Σ ω_k ℓ_k |
| `mhksa_train_adapter.py` | Stage 1: β_LAP, α̃ 학습 |
| `mhksa_stats.py` | Stage 2: source stats 저장 (`--cosine` 옵션) |
| `mhksa_adapt.py` | Stage 3: TTA (BN affine 업데이트) + 평가 (`--no_omega` ablation) |

하이퍼파라미터(H, M, K_W, P, K, N, L)는 §13 표 값을 dataset별 config에 넣어둠.

## 계획서에 없어서 임의로 정한 부분
1. **Stage 1 objective**: 계획서엔 "Adapter pretrain"만 있음 → backbone frozen, g 위에 임시 linear head를 붙여 source label MSE로 β_LAP, α̃ 학습 (head는 버림). epochs=5, lr=1e-2.
2. **m_h 샘플링**: "log-uniform"을 랜덤 샘플로 해석 (`bandwidth_mode: random`). 등간격은 `logspace`.
3. **Per-head / final PCA 공분산**: §7·§8 식은 1/N_s지만 §0 규약(unbiased)을 따라 1/(N_s−1)로 통일. §9 target batch 분산은 식 그대로 1/N_B.
4. **Wiener 흡수**: W_h^new(M×D)를 만들지 않고 U_g 좌표(K_W차원)에서 계산 — 수학적으로 동일 (수치 검증 오차 ~1e-8), D=16384에서도 메모리 부담 없음.
5. **MHKSA-cosine**: Wiener 후 g를 L2 정규화한 뒤 RFF (원논문 CoRP 대응). median distance도 정규화된 입력에서 계산. plain은 문서대로 원본 g에서.
6. **California(MLP)**: penultimate feature(C=100)를 [C,1,1] map으로 보고 N=2 → D=400 (§13의 D=400과 일치). 이 경우 LAP=GAP이라 β, α는 사실상 영향 없음.
7. **SVHN (S=2×2, N=4)**: cell이 전부 1×1이라 β_LAP가 효과 없음 (문서의 "degenerate"). H·P=48 < K=100이라 K=48로 자동 축소.
8. TTA 프로토콜(Adam lr 1e-3, batch 64, 1 epoch offline, fe_bn)은 SSA 공식 config를 그대로 사용.
