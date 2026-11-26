"""Pre-submit guard for the Track 2 repo rules. Run it before every push:

    python scripts/check_repo_rules.py           # checks the files that `git add .` would commit
    python scripts/check_repo_rules.py --head    # checks exactly what is committed at HEAD (what the judge clones)

Exit code 0 = all checks pass, 1 = at least one FAIL. Stdlib only, plus the organisers' scanner
(monsoon_r1.repo_scan, installed by requirements-train.txt) when it is available.

What it guards against:
  1. The organisers' repo scan raising any critical flag (archives, hardcoded dates, weather APIs, no checkpoint).
  2. Training / scripts / dashboard code being pulled into "inference code" (any .py that names the server route,
     and everything it imports, is treated as server code by the judge's scanner).
  3. The server importing a library the judge's image does not have, or touching the network / the data/ folder.
  4. Weather archives sneaking in: data files, big arrays, or a dashboard JSON that grows past a few demo windows.
  5. Repo size (2 GB limit incl. LFS) and files GitHub would reject without LFS.
  6. Python in dashboard/ (the dashboard must be a separate JS/TS app).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Same pattern the organisers' scanner uses to find the server file. Built in pieces so THIS file never matches it.
_ROUTE = "simu" + "late"
ROUTE_RE = re.compile(r"[/'\"]" + _ROUTE + r"['\"]|/" + _ROUTE + r"\b")

# Where inference code may live. Everything else (training/, scripts/, dashboard/) must stay out of the server graph.
INFERENCE_ALLOWED = re.compile(r"^(serve\.py|monsoon/.*\.py)$")

# Third-party packages available in the judge's sandbox (problem statement), plus the stdlib.
ALLOWED_THIRD_PARTY = {"torch", "numpy", "scipy", "pandas", "xarray", "sklearn", "lightgbm", "onnxruntime",
                       "fastapi", "flask", "uvicorn", "starlette", "pydantic"}
# Things the offline server must never do.
NETWORK_MODULES = {"requests", "httpx", "urllib", "urllib3", "aiohttp", "socket", "http", "ftplib", "modal",
                   "cdsapi", "gcsfs", "fsspec", "huggingface_hub"}
FORBIDDEN_TEXT_IN_SERVER = [r"\bdata/", r"india_climate", r"\.nc\b", r"chunks/", r"demo_windows", r"/home/", r"open_dataset"]

DATA_EXT = {".nc", ".nc4", ".grib", ".grib2", ".zarr", ".h5", ".hdf5", ".parquet", ".feather", ".csv"}
MAX_DASHBOARD_JSON_DAYS = 200        # demo windows: 13 days each; anything near a year of days looks like an archive
REPO_LIMIT_BYTES = 2 * 1024 ** 3
GITHUB_HARD_LIMIT = 100 * 1024 ** 2  # GitHub rejects plain files above this; use Git LFS

results: list[tuple[str, str, str]] = []   # (status, check, detail)


def report(status: str, check: str, detail: str = "") -> None:
    results.append((status, check, detail))


# ---- which files are we checking -----------------------------------------------------------------------------
def snapshot(head: bool) -> Path:
    """Copy the files the judge would get into a temp dir and return it."""
    tmp = Path(tempfile.mkdtemp(prefix="repo_rules_"))
    if head:
        data = subprocess.run(["git", "archive", "HEAD"], cwd=ROOT, check=True, capture_output=True).stdout
        tar = tmp / "_head.tar"
        tar.write_bytes(data)
        with tarfile.open(tar) as t:
            t.extractall(tmp, filter="data")
        tar.unlink()
    else:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT, check=True,
                             capture_output=True).stdout.decode()
        for rel in filter(None, out.split("\0")):
            src = ROOT / rel
            if src.is_file():
                (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, tmp / rel)
    return tmp


def files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and ".git" not in p.relative_to(root).parts and "node_modules" not in p.relative_to(root).parts:
            yield p


# ---- 1. organisers' scanner ----------------------------------------------------------------------------------
def check_official_scan(root: Path) -> set[str] | None:
    try:
        from monsoon_r1.repo_scan import scan
    except ImportError:
        report("WARN", "official repo scan", "monsoon_r1 not installed (pip install -r requirements-train.txt); skipped")
        return None
    r = scan(root)
    crit = [f for f in r["flags"] if f["level"] == "critical"]
    warns = [f for f in r["flags"] if f["level"] == "warn"]
    for f in crit:
        report("FAIL", "official scan: critical flag", f"{f['what']}" + (f" ({f['file']})" if "file" in f else ""))
    for f in warns:
        report("FAIL", "official scan: warning (costs points, goes to a human)",
               f"{f['what']}" + (f" ({f['file']})" if "file" in f else ""))
    if not crit and not warns:
        report("PASS", "official scan", f"{r['points']}/15 pts, health={r['health']}")
    else:
        report("INFO", "official scan score", f"{r['points']}/15 pts, health={r['health']}")
    return set(r["inference_files"])


# ---- 2/3. inference code graph -------------------------------------------------------------------------------
def _local_imports(src: str, here: Path, root: Path) -> set[Path]:
    """Same resolution rule as the organisers' scanner."""
    out: set[Path] = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    names: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            names.append(n.module); names += [f"{n.module}.{a.name}" for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.level:
            names += [a.name for a in n.names]
    for nm in names:
        parts = nm.split(".")
        for base in (here.parent, root):
            for k in range(len(parts), 0, -1):
                for cand in (base.joinpath(*parts[:k]).with_suffix(".py"), base.joinpath(*parts[:k], "__init__.py")):
                    if cand.is_file() and root.resolve() in cand.resolve().parents:
                        out.add(cand.resolve())
    return out


def inference_graph(root: Path) -> set[Path]:
    seeds = [p.resolve() for p in files(root) if p.suffix == ".py" and ROUTE_RE.search(p.read_text(errors="ignore"))]
    seen: set[Path] = set(); stack = list(seeds)
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack += list(_local_imports(p.read_text(errors="ignore"), p, root))
    return seen


def check_inference(root: Path) -> None:
    inf = inference_graph(root)
    rels = sorted(str(p.relative_to(root.resolve())) for p in inf)
    seeds = [r for r in rels if ROUTE_RE.search((root / r).read_text(errors="ignore"))]
    if seeds != ["serve.py"]:
        report("FAIL", "only serve.py names the server route", f"files treated as the server: {seeds}")
    else:
        report("PASS", "only serve.py names the server route")
    outside = [r for r in rels if not INFERENCE_ALLOWED.match(r)]
    if outside:
        report("FAIL", "inference code stays in serve.py + monsoon/", f"pulled in by imports: {outside}")
    else:
        report("PASS", "inference code stays in serve.py + monsoon/", ", ".join(rels))

    stdlib = set(sys.stdlib_module_names)
    local_top = {p.name for p in root.iterdir() if p.is_dir()} | {p.stem for p in root.glob("*.py")}
    for r in rels:
        src = (root / r).read_text(errors="ignore")
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            report("FAIL", "server code parses", f"{r}: {e}")
            continue
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
                mods.add(n.module.split(".")[0])
        net = sorted(mods & NETWORK_MODULES)
        if net:
            report("FAIL", "server makes no network calls", f"{r} imports {net}")
        missing = sorted(m for m in mods if m not in stdlib and m not in local_top and m not in ALLOWED_THIRD_PARTY)
        if missing:
            report("FAIL", "server only uses libraries in the judge's image", f"{r} imports {missing}")
        for pat in FORBIDDEN_TEXT_IN_SERVER:
            if re.search(pat, src):
                report("FAIL", "server never reads training data", f"{r} mentions {pat!r}")
    if not any(s == "FAIL" and c.startswith("server") for s, c, _ in results):
        report("PASS", "server imports: judge-image libs only, no network, no data/ reads")


# ---- 4. archives ---------------------------------------------------------------------------------------------
def _json_days(obj, depth=0) -> int:
    """Count [6][32][36]-shaped days anywhere in a JSON document."""
    if depth > 8 or not isinstance(obj, (list, dict)):
        return 0
    if isinstance(obj, dict):
        return sum(_json_days(v, depth + 1) for v in obj.values())
    if (len(obj) == 6 and all(isinstance(c, list) and len(c) == 32 and isinstance(c[0], list) and len(c[0]) == 36
                              for c in obj)):
        return 1
    return sum(_json_days(v, depth + 1) for v in obj)


def check_archives(root: Path) -> None:
    bad = [str(p.relative_to(root)) for p in files(root) if p.suffix.lower() in DATA_EXT]
    if bad:
        report("FAIL", "no climate data files committed", ", ".join(bad))
    else:
        report("PASS", "no climate data files committed")

    total = 0
    for p in files(root):
        if p.suffix != ".json" or p.stat().st_size < 200_000:
            continue
        try:
            days = _json_days(json.loads(p.read_text()))
        except Exception:
            continue
        total += days
        if days:
            report("INFO", "weather days stored in JSON", f"{p.relative_to(root)}: {days} days")
    if total > MAX_DASHBOARD_JSON_DAYS:
        report("FAIL", "JSON weather stays demo-sized", f"{total} days in total > {MAX_DASHBOARD_JSON_DAYS}: looks like an archive")
    else:
        report("PASS", "JSON weather stays demo-sized", f"{total} days in total (limit {MAX_DASHBOARD_JSON_DAYS})")

    py_in_dash = [str(p.relative_to(root)) for p in files(root) if p.suffix == ".py" and p.relative_to(root).parts[0] == "dashboard"]
    if py_in_dash:
        report("FAIL", "dashboard is JS/TS only", ", ".join(py_in_dash))
    else:
        report("PASS", "dashboard is JS/TS only")


# ---- 5. size -------------------------------------------------------------------------------------------------
def check_size(root: Path) -> None:
    lfs_ptr = re.compile(rb"^version https://git-lfs.github.com/spec/v1\noid sha256:\w+\nsize (\d+)", re.M)
    total = 0; too_big = []
    lfs_tracked = set()
    try:
        out = subprocess.run(["git", "lfs", "ls-files", "-n"], cwd=ROOT, capture_output=True, text=True)
        lfs_tracked = set(out.stdout.split()) if out.returncode == 0 else set()
    except FileNotFoundError:
        pass
    for p in files(root):
        rel = str(p.relative_to(root)); size = p.stat().st_size
        head = p.open("rb").read(200)
        m = lfs_ptr.match(head)
        if m:
            size = int(m.group(1))
        elif size > GITHUB_HARD_LIMIT and rel not in lfs_tracked:
            too_big.append(f"{rel} ({size / 1e6:.0f} MB)")
        total += size
    if too_big:
        report("FAIL", "files over 100 MB use Git LFS", ", ".join(too_big))
    else:
        report("PASS", "files over 100 MB use Git LFS")
    status = "PASS" if total < REPO_LIMIT_BYTES else "FAIL"
    report(status, "repo size incl. LFS < 2 GB", f"{total / 1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", action="store_true", help="check HEAD (what the judge clones) instead of the working tree")
    a = ap.parse_args()
    root = snapshot(a.head)
    try:
        check_official_scan(root)
        check_inference(root)
        check_archives(root)
        check_size(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    width = max(len(c) for _, c, _ in results)
    for s, c, d in results:
        print(f"[{s:4}] {c:<{width}}  {d}")
    fails = sum(s == "FAIL" for s, _, _ in results)
    print(f"\n{'FAIL' if fails else 'OK'}: {fails} failing check(s) on {'HEAD' if a.head else 'working tree'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
