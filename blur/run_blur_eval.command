#!/bin/bash
# Double-click me (or: bash blur/run_blur_eval.command). Runs the shared TrOCR
# evaluation on every image in blur/inputs and prints CER/WER per condition.
cd "$(dirname "$0")/.." || exit 1
mkdir -p blur/outputs
LOG=blur/outputs/eval_run.log
{
  echo "=== $(date) start"
  if ! command -v uv >/dev/null 2>&1; then
    if [ -x "$HOME/.local/bin/uv" ]; then export PATH="$HOME/.local/bin:$PATH";
    else echo "uv is not installed. Install it first:  curl -LsSf https://astral.sh/uv/install.sh | sh"; echo "=== FAILED: no uv"; exit 1; fi
  fi
  uv --version
  uv run main.py blur && uv run blur/summarize_by_condition.py --csv blur/outputs/by_condition.csv
  echo "=== $(date) done (exit $?)"
} 2>&1 | tee "$LOG"
echo; echo "Finished. Results: blur/outputs/summary.json, blur/outputs/by_condition.csv"
read -r -p "Press Enter to close this window..." _
