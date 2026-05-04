#!/usr/bin/env bash
# Stage-1 pretraining for NAST on IWSLT14 DE-EN.
# Effective batch = 1 GPU * 32768 max_tokens * 2 update_freq = 64k tokens.
# Tuned for Blackwell 96GB: 32k single-pass + 2x grad-accum hits ~75k wps,
# ~0.77 sec/update; 150k updates in ~32 h.
# Auto-resumes from checkpoint_last.pt if present.
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK=$PROJECT/work
PLUGIN=$PROJECT/NAST
DATA_BIN=$WORK/data-bin/iwslt14.de-en.joined
EXP=${EXP:-iwslt14_s1}
CKPT=$WORK/checkpoints/$EXP
WAIT_UNTIL=${WAIT_UNTIL:-0}

[[ -d "$DATA_BIN" ]] || { echo "DATA_BIN not found: $DATA_BIN" >&2; exit 1; }
[[ -d "$PLUGIN" ]] || { echo "PLUGIN not found: $PLUGIN" >&2; exit 1; }
[[ "$WAIT_UNTIL" =~ ^[0-9]+$ ]] || { echo "WAIT_UNTIL must be a non-negative integer (got: $WAIT_UNTIL)" >&2; exit 1; }
mkdir -p "$CKPT" "$WORK/logs"
cd "$WORK"

# CUDA arch default 12.0 = Blackwell sm_120 (RTX PRO 6000). Override via env on other GPUs.
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-/usr/local/cuda}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

LOG=$WORK/logs/$EXP.log
echo "[$(date '+%F %T')] Stage-1 start: EXP=$EXP WAIT_UNTIL=$WAIT_UNTIL" | tee -a "$LOG"

fairseq-train "$DATA_BIN" \
    --user-dir "$PLUGIN" \
    --fp16 \
    --save-dir "$CKPT" \
    --ddp-backend=legacy_ddp \
    --task translation_ctc_streaming \
    --criterion nat_loss_ngram_glat_simul --left-pad-source --glat-p 0.5:0.3@100k \
    --src-embedding-copy \
    --src-upsample-ratio 3 --plain-ctc --wait-until "$WAIT_UNTIL" --latency-factor 0 \
    --arch nonautoregressive_streaming_transformer_iwslt \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 4000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.01 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --num-workers 4 \
    --max-tokens 32768 \
    --update-freq 2 \
    --save-interval-updates 2000 \
    --keep-interval-updates 10 --keep-last-epochs 5 \
    --max-update 150000 2>&1 | tee -a "$LOG"
