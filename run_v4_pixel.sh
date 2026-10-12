#!/usr/bin/env bash
set -Eeuo pipefail

# One entry point: train a separate V4 Pixel relation head for each dataset,
# diagnose the held-out test split, and export test + train segment predictions.
# CORD dev/valid is intentionally never read.

REPO="${REPO:-/home/s24gbn1/Documents/httn/unilm/layoutlmv3}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
DATASET="${1:-all}"             # all | funsd | cord
EPOCHS="${EPOCHS:-15}"
SEED="${SEED:-1993}"
# Keep the original V4 Pixel inference settings as defaults; override only if
# choosing values from internal validation (never tune on test scores).
THRESHOLD="${THRESHOLD:-0.50}"
EDGE_TOP_K="${EDGE_TOP_K:-16}"
PIXEL_GATE="${PIXEL_GATE:-0.12}"
PIXEL_ORDER_WEIGHT="${PIXEL_ORDER_WEIGHT:-0.10}"
OUT_ROOT="${OUT_ROOT:-$REPO/segment_outputs}"
V4_SCRIPT="${V4_SCRIPT:-$REPO/segment_v4_pixel.py}"
if [[ "$V4_SCRIPT" != /* ]]; then V4_SCRIPT="$REPO/$V4_SCRIPT"; fi

# Keep the runner compatible with both known model locations.
MODEL_PATH="${MODEL_PATH:-$REPO/models/layoutlmv3-base}"
if [[ "$MODEL_PATH" != /* ]]; then MODEL_PATH="$REPO/$MODEL_PATH"; fi
if [[ ! -f "$MODEL_PATH/config.json" && -f "/home/s24gbn1/Documents/phg/unilm/layoutlmv3/models/layoutlmv3-base/config.json" ]]; then
  MODEL_PATH="/home/s24gbn1/Documents/phg/unilm/layoutlmv3/models/layoutlmv3-base"
fi

FUNSD_ROOT="$REPO/datasets/funsd/dataset"
CORD_ROOT="${CORD_ROOT:-/home/s24gbn1/Documents/phg/unilm/layoutlmv3/CORD}"
if [[ ! -d "$CORD_ROOT/train/image" && -d "$REPO/CORD/train/image" ]]; then
  CORD_ROOT="$REPO/CORD"
fi

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

fail() { echo "ERROR: $*" >&2; exit 2; }

[[ -d "$REPO" ]] || fail "Repository not found: $REPO"
[[ -f "$V4_SCRIPT" ]] || fail "V4 script not found: $V4_SCRIPT. Copy the supplied segment_v4_pixel.py into the repository first."
[[ -f "$MODEL_PATH/config.json" ]] || fail "LayoutLMv3 config not found: $MODEL_PATH/config.json. Set MODEL_PATH=/absolute/path/to/models/layoutlmv3-base"

cd "$REPO"
"$PYTHON" -m py_compile "$V4_SCRIPT"

check_data() {
  local name="$1" images="$2" annotations="$3" fmt="$4"
  "$PYTHON" - "$name" "$images" "$annotations" "$fmt" <<'PY'
import json, sys
from pathlib import Path
name, image_dir, ann_dir, fmt = sys.argv[1:]
image_dir, ann_dir = Path(image_dir), Path(ann_dir)
exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
if not image_dir.is_dir():
    raise SystemExit("{} image directory missing: {}".format(name, image_dir))
if not ann_dir.is_dir():
    raise SystemExit("{} annotation directory missing: {}".format(name, ann_dir))
anns = sorted(ann_dir.glob("*.json"))
if not anns:
    raise SystemExit("{} has no JSON annotations in {}".format(name, ann_dir))
missing = []
for ann in anns:
    if not any((image_dir / (ann.stem + ext)).is_file() for ext in exts):
        missing.append(ann.stem)
if missing:
    raise SystemExit("{}: {} JSON files lack matching images, e.g. {}".format(name, len(missing), missing[:8]))
# Validate schema on every annotation before beginning an expensive GPU run.
for ann in anns:
    try:
        with ann.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise SystemExit("{} invalid JSON {}: {}".format(name, ann, exc))
    required = "form" if fmt == "funsd" else "valid_line"
    if required not in data or not isinstance(data[required], list):
        raise SystemExit("{} schema mismatch in {}: expected list key {!r}".format(name, ann, required))
    if fmt == "cord":
        for line_idx, line in enumerate(data["valid_line"]):
            if not isinstance(line, dict) or not isinstance(line.get("words", []), list):
                raise SystemExit("{} invalid valid_line[{}] in {}".format(name, line_idx, ann))
            for word_idx, word in enumerate(line.get("words", [])):
                if str(word.get("text", "")).strip():
                    quad = word.get("quad")
                    if not isinstance(quad, dict) or not all(k in quad for k in ("x1", "y1", "x3", "y3")):
                        raise SystemExit("{} invalid CORD quad at valid_line[{}].words[{}] in {}".format(name, line_idx, word_idx, ann))
    else:
        for form_idx, form in enumerate(data["form"]):
            if not isinstance(form, dict) or not isinstance(form.get("words", []), list):
                raise SystemExit("{} invalid form[{}] in {}".format(name, form_idx, ann))
            for word_idx, word in enumerate(form.get("words", [])):
                if str(word.get("text", "")).strip():
                    box = word.get("box")
                    if not isinstance(box, (list, tuple)) or len(box) != 4:
                        raise SystemExit("{} invalid FUNSD box at form[{}].words[{}] in {}".format(name, form_idx, word_idx, ann))
print("[PREFLIGHT] {}: {} matched image/annotation pairs; schema={}".format(name, len(anns), fmt))
PY
}

run_funsd() {
  local train_img="$FUNSD_ROOT/training_data/images"
  local train_ann="$FUNSD_ROOT/training_data/annotations"
  local test_img="$FUNSD_ROOT/testing_data/images"
  local test_ann="$FUNSD_ROOT/testing_data/annotations"
  local out="$OUT_ROOT/funsd"
  local head="$out/segment_head_v4_pixel.pt"

  echo
  echo "==================== FUNSD / V4 PIXEL ===================="
  check_data "FUNSD train" "$train_img" "$train_ann" funsd
  check_data "FUNSD test" "$test_img" "$test_ann" funsd
  mkdir -p "$out"
  # Remove just the target dataset's head so a failed training run cannot fall back
  # to a stale checkpoint from an earlier run.
  rm -f "$head"

  "$PYTHON" "$V4_SCRIPT" train \
    --model-path "$MODEL_PATH" --device "$DEVICE" --dataset-format funsd \
    --train-images "$train_img" --train-annotations "$train_ann" \
    --epochs "$EPOCHS" --lr 2e-4 --weight-decay 1e-2 \
    --negative-ratio 3 --max-positive-per-page 4000 --batch-size 1024 \
    --top-k 64 --pixel-top-k 24 --seed "$SEED" --head-out "$head"

  "$PYTHON" "$V4_SCRIPT" diagnose \
    --model-path "$MODEL_PATH" --device "$DEVICE" --dataset-format funsd \
    --head "$head" --images "$test_img" --annotations "$test_ann" \
    --top-k 64 --pixel-top-k 24 --output "$out/diagnostic_v4_pixel"

  # Original V4 Pixel clustering settings are intentionally retained.
  "$PYTHON" "$V4_SCRIPT" predict \
    --model-path "$MODEL_PATH" --device "$DEVICE" --dataset-format funsd \
    --head "$head" --images "$test_img" --annotations "$test_ann" \
    --top-k 64 --pixel-top-k 24 --threshold "$THRESHOLD" --edge-top-k "$EDGE_TOP_K" \
    --pixel-gate "$PIXEL_GATE" --pixel-order-weight "$PIXEL_ORDER_WEIGHT" --evaluate \
    --output "$out/v4_test"

  "$PYTHON" "$V4_SCRIPT" predict \
    --model-path "$MODEL_PATH" --device "$DEVICE" --dataset-format funsd \
    --head "$head" --images "$train_img" --annotations "$train_ann" \
    --top-k 64 --pixel-top-k 24 --threshold "$THRESHOLD" --edge-top-k "$EDGE_TOP_K" \
    --pixel-gate "$PIXEL_GATE" --pixel-order-weight "$PIXEL_ORDER_WEIGHT" \
    --output "$out/v4_train"

  echo "[DONE] FUNSD head: $head"
  echo "[DONE] FUNSD test: $out/v4_test"
  echo "[DONE] FUNSD train: $out/v4_train"
}

run_cord() {
  local train_img="$CORD_ROOT/train/image"
  local train_ann="$CORD_ROOT/train/json"
  local test_img="$CORD_ROOT/test/image"
  local test_ann="$CORD_ROOT/test/json"
  local out="$OUT_ROOT/cord"
  local head="$out/segment_head_v4_pixel.pt"

  echo
  echo "==================== CORD / V4 PIXEL ====================="
  check_data "CORD train" "$train_img" "$train_ann" cord
  check_data "CORD test" "$test_img" "$test_ann" cord
  mkdir -p "$out"
  # Remove just the target dataset's head so a failed training run cannot fall back
  # to a stale checkpoint from an earlier run.
  rm -f "$head"

  "$PYTHON" "$V4_SCRIPT" train \
    --model-path "$MODEL_PATH" --device "$DEVICE" --dataset-format cord \
    --train-images "$train_img" --train-annotations "$train_ann" \
    --epochs "$EPOCHS" --lr 2e-4 --weight-decay 1e-2 \
    --negative-ratio 3 --max-positive-per-page 4000 --batch-size 1024 \
    --top-k 64 --pixel-top-k 24 --seed "$SEED" --head-out "$head"

  "$PYTHON" "$V4_SCRIPT" diagnose \
    --model-path "$MODEL_PATH" --device "$DEVICE" --dataset-format cord \
    --head "$head" --images "$test_img" --annotations "$test_ann" \
    --top-k 64 --pixel-top-k 24 --output "$out/diagnostic_v4_pixel"

  "$PYTHON" "$V4_SCRIPT" predict \
    --model-path "$MODEL_PATH" --device "$DEVICE" --dataset-format cord \
    --head "$head" --images "$test_img" --annotations "$test_ann" \
    --top-k 64 --pixel-top-k 24 --threshold "$THRESHOLD" --edge-top-k "$EDGE_TOP_K" \
    --pixel-gate "$PIXEL_GATE" --pixel-order-weight "$PIXEL_ORDER_WEIGHT" --evaluate \
    --output "$out/v4_test"

  "$PYTHON" "$V4_SCRIPT" predict \
    --model-path "$MODEL_PATH" --device "$DEVICE" --dataset-format cord \
    --head "$head" --images "$train_img" --annotations "$train_ann" \
    --top-k 64 --pixel-top-k 24 --threshold "$THRESHOLD" --edge-top-k "$EDGE_TOP_K" \
    --pixel-gate "$PIXEL_GATE" --pixel-order-weight "$PIXEL_ORDER_WEIGHT" \
    --output "$out/v4_train"

  echo "[DONE] CORD head: $head"
  echo "[DONE] CORD test: $out/v4_test"
  echo "[DONE] CORD train: $out/v4_train"
  echo "[NOTE] CORD dev/valid was not accessed. Validation for training is split only from CORD train."
}

case "$DATASET" in
  funsd) run_funsd ;;
  cord) run_cord ;;
  all) run_funsd; run_cord ;;
  *) fail "Usage: bash run_v4_pixel.sh [all|funsd|cord]" ;;
esac

printf '\nALL REQUESTED DATASETS FINISHED\n'
printf 'Output root: %s\n' "$OUT_ROOT"
