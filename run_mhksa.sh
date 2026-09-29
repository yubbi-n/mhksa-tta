#!/bin/bash
# 사용법: bash run_mhksa.sh svhn            (configs/mhksa/svhn.yaml)
# 전제: SSA 공식 repo 루트에서 실행, source model은 train_source.py로 이미 학습돼 있어야 함
set -e
NAME=${1:-svhn}
CFG=configs/mhksa/${NAME}.yaml
OUT=result/mhksa/${NAME}

# Stage 1: LAGM adapter (β_LAP, α̃)
python3 mhksa_train_adapter.py -c $CFG -o $OUT

# Stage 2: source stats — plain / cosine 두 버전
python3 mhksa_stats.py -c $CFG -o $OUT
python3 mhksa_stats.py -c $CFG -o $OUT --cosine

# Stage 3: TTA (SSA 논문처럼 seed 3개)
for VAR in plain cosine; do
  for SEED in 0 1 2; do
    python3 mhksa_adapt.py -c $CFG --stats $OUT/mhksa_stats_${VAR}.pt \
      -o result/mhksa_tta/${NAME}_${VAR}/seed${SEED} --seed $SEED
  done
done
