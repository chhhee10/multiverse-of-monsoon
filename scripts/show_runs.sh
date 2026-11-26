#!/usr/bin/env bash
# Fetch metrics for Modal runs and print a compact table:  scripts/show_runs.sh v1 v1big ...
cd "$(dirname "$0")/.."
for r in "$@"; do
  mkdir -p "runs/$r"; rm -f "runs/$r/metrics.jsonl"
  .venv/bin/modal volume get monsoon-data "runs/$r/metrics.jsonl" "runs/$r/metrics.jsonl" --force >/dev/null 2>&1
  echo "== $r"
  .venv/bin/python - "runs/$r/metrics.jsonl" <<'PY'
import json, sys, math
try: lines = open(sys.argv[1]).read().splitlines()
except FileNotFoundError: print("  (no metrics yet)"); sys.exit()
for l in lines:
    r = json.loads(l)
    if not math.isfinite(r["train_mse"]): continue
    print(f"  {r['phase']:3s} it={r['it']:5d} {r['minutes']:5.1f}m mse={r['train_mse']:.4f} ss={r['mean_ss']:.4f} "
          f"csi_sk={r['mean_csi_skill']:.4f} rmse={r['rmse_pts']:4.1f} csi={r['csi_pts']:4.1f} blur={r['blur_pts']:4.1f} "
          f"{r['blur_ratio']} {'BEST' if r.get('best') else ''}")
PY
done
