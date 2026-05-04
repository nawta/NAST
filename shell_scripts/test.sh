#!/usr/bin/env bash
# Inference + BLEU + latency metrics for one trained checkpoint.
# Required env: AVG_CKPT (path to averaged checkpoint), EXP (label for log).
# Optional env: WAIT_UNTIL (default 0), GEN_SUBSET (default test)
set -euo pipefail

PROJECT="${PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
WORK=$PROJECT/work
PLUGIN=$PROJECT/NAST
DATA_BIN=$WORK/data-bin/iwslt14.de-en.joined
AVG_CKPT=${AVG_CKPT:?AVG_CKPT required}
EXP=${EXP:?EXP required}
WAIT_UNTIL=${WAIT_UNTIL:-0}
GEN_SUBSET=${GEN_SUBSET:-test}
REF_FILE=${REF_FILE:-$WORK/data/iwslt14.tok/${GEN_SUBSET}.en}

[[ -f "$AVG_CKPT" ]] || { echo "AVG_CKPT not found: $AVG_CKPT" >&2; exit 1; }
[[ -d "$DATA_BIN" ]] || { echo "DATA_BIN not found: $DATA_BIN" >&2; exit 1; }
[[ -f "$REF_FILE" ]] || { echo "REF_FILE not found: $REF_FILE" >&2; exit 1; }
[[ -f "$PROJECT/shell_scripts/multi-bleu.perl" ]] || { echo "multi-bleu.perl not found" >&2; exit 1; }
[[ "$WAIT_UNTIL" =~ ^[0-9]+$ ]] || { echo "WAIT_UNTIL must be a non-negative integer (got: $WAIT_UNTIL)" >&2; exit 1; }
mkdir -p "$WORK/logs"
cd "$WORK"

# CUDA arch default 12.0 = Blackwell sm_120 (RTX PRO 6000). Override via env on other GPUs.
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-/usr/local/cuda}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

OUT=$WORK/logs/eval_${EXP}.out
HYP=$WORK/logs/eval_${EXP}.hyp

# --model-overrides uses Python literals (True/False) because fairseq parses via eval.
python "$PLUGIN/scripts/generate_streaming.py" "$DATA_BIN" \
    --user-dir "$PLUGIN" \
    --gen-subset "$GEN_SUBSET" \
    --src-upsample-ratio 3 --plain-ctc \
    --wait-until "$WAIT_UNTIL" \
    --model-overrides "{\"wait_until\":${WAIT_UNTIL},\"src_upsample_ratio\":3,\"src_embedding_copy\":True,\"plain_ctc\":True}" \
    --task translation_ctc_streaming \
    --path "$AVG_CKPT" \
    --max-tokens 2048 --remove-bpe \
    --left-pad-source > "$OUT"

# fairseq 5175fd's generate_streaming.py emits D-lines as `D-<id>\t<hyp>` (2 cols),
# unlike newer fairseq which adds a score column. Use cut -f 2- accordingly.
# `|| true` keeps the pipeline alive under set -e so the empty-output check below fires.
grep '^D-' "$OUT" | sort -V | cut -f 2- > "$HYP" || true
[[ -s "$HYP" ]] || { echo "no D-lines extracted; generation likely failed" >&2; exit 1; }
echo "==== multi-bleu ($EXP, wait_until=$WAIT_UNTIL) ===="
perl "$PROJECT/shell_scripts/multi-bleu.perl" -lc "$REF_FILE" < "$HYP"
echo "==== latency ($EXP) ===="
grep -E '(CW|AP|AL|DAL) score:' "$OUT" || true
