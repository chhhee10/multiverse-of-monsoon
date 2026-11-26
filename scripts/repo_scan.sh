#!/usr/bin/env bash
# Run the organisers' repo scan (team kit, monsoon_r1.repo_scan) on exactly what is committed at HEAD,
# i.e. what the judge's clone will contain (git-ignored data/ and runs/ are not part of it).
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git archive HEAD | tar -x -C "$tmp"
.venv/bin/python - "$tmp" <<'EOF'
import json, sys
from monsoon_r1.repo_scan import scan
r = scan(sys.argv[1])
print(json.dumps({k: r[k] for k in ("points", "anti_cheat_points", "code_health_points", "flags", "health", "inference_files")}, indent=1))
ok = r["inference_files"] == ["serve.py"] or all(not f.startswith("training/") for f in r["inference_files"])
if not ok:
    print("FAIL: training code is being treated as inference code", file=sys.stderr); sys.exit(1)
EOF
