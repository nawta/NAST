#!/usr/bin/env bash
# Stage-2 latency finetune for NAST on IWSLT14 DE-EN.
# Effective batch = 1 GPU * 16384 max_tokens * 4 update_freq = 64k tokens.
# Note: paper used 256k for Stage-2; smoke test shows gnorm stable at 64k,
# so we trade slight stochasticity for ~4x throughput (~2.2 h per point).
# Auto-resumes from checkpoint_last.pt.
# Required env: PRETRAIN, EXP, WAIT_UNTIL, LATENCY_FACTOR, LATENCY_THRESHOLD
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK=$PROJECT/work
PLUGIN=$PROJECT/NAST
DATA_BIN=$WORK/data-bin/iwslt14.de-en.joined
EXP=${EXP:?EXP env var required (e.g. iwslt14_s2_p3)}
CKPT=$WORK/checkpoints/$EXP
PRETRAIN=${PRETRAIN:?PRETRAIN env var required (path to Stage-1 avg5.pt or checkpoint_best.pt)}
WAIT_UNTIL=${WAIT_UNTIL:?WAIT_UNTIL required}
LATENCY_FACTOR=${LATENCY_FACTOR:?LATENCY_FACTOR required}
LATENCY_THRESHOLD=${LATENCY_THRESHOLD:?LATENCY_THRESHOLD required}

[[ -d "$DATA_BIN" ]] || { echo "DATA_BIN not found: $DATA_BIN" >&2; exit 1; }
[[ -f "$PRETRAIN" ]] || { echo "PRETRAIN ckpt not found: $PRETRAIN" >&2; exit 1; }
[[ -d "$PLUGIN" ]] || { echo "PLUGIN not found: $PLUGIN" >&2; exit 1; }
[[ "$WAIT_UNTIL" =~ ^[0-9]+$ ]] || { echo "WAIT_UNTIL must be a non-negative integer (got: $WAIT_UNTIL)" >&2; exit 1; }
[[ "$LATENCY_FACTOR"    =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "LATENCY_FACTOR must be a non-negative number (got: $LATENCY_FACTOR)" >&2; exit 1; }
[[ "$LATENCY_THRESHOLD" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "LATENCY_THRESHOLD must be a non-negative number (got: $LATENCY_THRESHOLD)" >&2; exit 1; }
mkdir -p "$CKPT" "$WORK/logs"
cd "$WORK"

# CUDA arch default 12.0 = Blackwell sm_120 (RTX PRO 6000). Override via env on other GPUs.
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-/usr/local/cuda}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

LOG=$WORK/logs/$EXP.log
echo "[$(date '+%F %T')] Stage-2 start: EXP=$EXP WAIT_UNTIL=$WAIT_UNTIL LATENCY=($LATENCY_FACTOR,$LATENCY_THRESHOLD)" | tee -a "$LOG"

fairseq-train "$DATA_BIN" \
    --user-dir "$PLUGIN" \
    --fp16 \
    --finetune-from-model "$PRETRAIN" \
    --save-dir "$CKPT" \
    --ddp-backend=legacy_ddp \
    --task translation_ctc_streaming \
    --criterion nat_loss_ngram_glat_simul --left-pad-source --glat-p 0.3:0.3@8k \
    --src-embedding-copy \
    --src-upsample-ratio 3 --plain-ctc --wait-until "$WAIT_UNTIL" \
    --latency-factor "$LATENCY_FACTOR" --latency-threshold "$LATENCY_THRESHOLD" \
    --arch nonautoregressive_streaming_transformer_iwslt \
    --use-ngram --ngram-size 2 \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0003 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 500 \
    --warmup-init-lr '1e-07' \
    --dropout 0.1 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --log-format 'simple' --log-interval 10 \
    --fixed-validation-seed 7 \
    --num-workers 4 \
    --max-tokens 16384 \
    --update-freq 4 \
    --save-interval-updates 500 \
    --keep-interval-updates 10 --keep-last-epochs 3 \
    --max-update 8000 2>&1 | tee -a "$LOG"
