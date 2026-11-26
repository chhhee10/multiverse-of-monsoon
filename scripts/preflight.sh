#!/usr/bin/env bash
# Preflight: what the automated judge checks, run locally on exactly what is committed at HEAD.
#   1. organisers' repo scan on the committed tree
#   2. server in a judge-like container: no network, 4 CPUs, 6 GB RAM, read-only repo, writable /tmp only
#   3. organisers' schema validator against it (second container sharing the server's network namespace)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 1. repo scan"
scripts/repo_scan.sh

echo "== 2. judge-like container"
docker build -q -t monsoon-judge -f scripts/judge.Dockerfile scripts >/dev/null
docker rm -f monsoon-judge >/dev/null 2>&1 || true
tmp=$(mktemp -d)
trap 'docker rm -f monsoon-judge >/dev/null 2>&1 || true; rm -rf "$tmp"' EXIT
git archive HEAD | tar -x -C "$tmp"
t0=$(date +%s)
docker run -d --name monsoon-judge --network none --cpus 4 --memory 6g --read-only --tmpfs /tmp -e HOME=/tmp \
  -v "$tmp":/sub:ro -w /sub monsoon-judge python serve.py --port 8000 >/dev/null

echo "== 3. schema validator"
docker run --rm --network container:monsoon-judge monsoon-judge python -m monsoon_r1.schema_validator http://127.0.0.1:8000
echo "validator finished $(( $(date +%s) - t0 )) s after container start"
echo "== server log (should be empty)"
docker logs monsoon-judge 2>&1 | tail -5
