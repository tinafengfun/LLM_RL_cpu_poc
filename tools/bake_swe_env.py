#!/usr/bin/env python3
"""T24: bake SWE-bench (sympy subset) environment on node2 ← design 09 section 6.

Subcommands:
  fetch   pull SWE-bench_Verified rows via HF datasets-server JSON API,
          filter to one repo, write instances.jsonl
  clone   git clone --mirror the upstream repo (once, resumable by hand)
  venv    build a python venv with test deps, pack as tarball
  verify  self-check: extract venv + worktree one base_commit + collect tests
  all     fetch -> clone -> venv -> verify

Pure helpers (rows_to_instances / select_subset / page_url / build_manifest)
are unit-tested on the dev box; the heavy steps run on node2 (has internet).

Layout produced under --root (default /workspace/swebench):
  instances.jsonl            subset rows (SWE-bench schema)
  repo/<name>.git            bare mirror with all commits
  envs/<tag>.tar.gz          venv tarball (pytest + test deps)
  environments.json          manifest
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request

HF_API = "https://datasets-server.huggingface.co/rows"
DATASET = "princeton-nlp/SWE-bench_Verified"


# ---------------------------------------------------------------------------
# pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def page_url(offset: int, length: int, dataset: str = DATASET,
             config: str = "default", split: str = "test") -> str:
    ds = urllib.parse.quote(dataset, safe="")
    return (f"{HF_API}?dataset={ds}&config={config}&split={split}"
            f"&offset={offset}&length={length}")


def rows_to_instances(payload: dict) -> list[dict]:
    return [r["row"] for r in payload.get("rows", []) if "row" in r]


def parse_version(v: str) -> tuple:
    parts = [int(x) for x in str(v).split(".")[:2] if x.isdigit()]
    return tuple(parts) if parts else (0,)


def select_subset(instances: list[dict], repo: str,
                  limit: int | None = None,
                  min_version: str | None = None) -> list[dict]:
    out = [r for r in instances if r.get("repo") == repo]
    if min_version is not None:
        floor = parse_version(min_version)
        out = [r for r in out if parse_version(r.get("version", "0")) >= floor]
    out.sort(key=lambda r: str(r.get("instance_id", "")))
    if limit is not None:
        out = out[:limit]
    if not out:
        raise ValueError(f"no instances for repo={repo} min_version={min_version}")
    return out


def build_manifest(root: str, repo: str, git_dir: str, venv_tar: str,
                   venv_tag: str, instances_path: str, count: int) -> dict:
    return {
        "root": root,
        "repo": repo,
        "git_dir": git_dir,
        "venv_tarball": venv_tar,
        "venv_tag": venv_tag,
        "instances": instances_path,
        "num_instances": count,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# ---------------------------------------------------------------------------
# node2 execution steps
# ---------------------------------------------------------------------------

def _get(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "rl_sim-bake/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def step_fetch(root: str, repo: str, limit: int,
               min_version: str | None = None) -> str:
    os.makedirs(root, exist_ok=True)
    rows: list[dict] = []
    offset, page = 0, 100
    total = None
    while total is None or offset < total:
        payload = _get(page_url(offset, page))
        total = payload.get("num_rows_total", 0)
        rows.extend(rows_to_instances(payload))
        offset += page
        print(f"[fetch] {offset}/{total}")
    subset = select_subset(rows, repo, limit, min_version)
    path = os.path.join(root, "instances.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in subset:
            f.write(json.dumps(r) + "\n")
    print(f"[fetch] wrote {len(subset)} instances -> {path}")
    return path


def step_clone(root: str, repo: str) -> str:
    name = repo.split("/")[-1]
    git_dir = os.path.join(root, "repo", f"{name}.git")
    if os.path.isdir(git_dir):
        print(f"[clone] exists: {git_dir}")
        return git_dir
    os.makedirs(os.path.dirname(git_dir), exist_ok=True)
    subprocess.run(["git", "clone", "--mirror",
                    f"https://github.com/{repo}", git_dir], check=True)
    print(f"[clone] done -> {git_dir}")
    return git_dir


def step_venv(root: str, tag: str, deps: list[str]) -> str:
    env_dir = os.path.join(root, "envs", tag)
    tar_path = os.path.join(root, "envs", f"{tag}.tar.gz")
    if os.path.exists(tar_path):
        print(f"[venv] exists: {tar_path}")
        return tar_path
    subprocess.run([sys.executable, "-m", "venv", env_dir], check=True)
    pip = os.path.join(env_dir, "bin", "pip")
    subprocess.run([pip, "install", "-q", "--disable-pip-version-check",
                    *deps], check=True)
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(env_dir, arcname=tag)
    print(f"[venv] packed -> {tar_path}")
    return tar_path


def step_verify(root: str, repo: str, venv_tag: str = "py311") -> None:
    inst_path = os.path.join(root, "instances.jsonl")
    git_dir = os.path.join(root, "repo", f"{repo.split('/')[-1]}.git")
    with open(inst_path, encoding="utf-8") as f:
        inst = json.loads(f.readline())
    commit = inst["base_commit"]
    # The venv is used IN PLACE (read-only shared layer, DSec-style): venv
    # scripts carry build-path shebangs, so relocating via tarball breaks them.
    venv = os.path.join(root, "envs", venv_tag)
    with tempfile.TemporaryDirectory(prefix="bake_verify_") as td:
        worktree = os.path.join(td, "wt")
        subprocess.run(["git", "--git-dir", git_dir, "worktree", "add",
                        "--detach", worktree, commit], check=True,
                       capture_output=True)
        py = os.path.join(venv, "bin", "python")
        env = dict(os.environ, PYTHONPATH=worktree)
        subprocess.run([py, "-c", "import sympy; print('sympy', sympy.__version__)"],
                       check=True, env=env, cwd=worktree)
        f2p = json.loads(inst["FAIL_TO_PASS"])
        node = f2p[0].split("::")[0]
        r = subprocess.run([py, "-m", "pytest", node, "--collect-only", "-q"],
                           env=env, cwd=worktree, capture_output=True, text=True)
        if r.returncode not in (0, 5):  # 5 = no tests collected
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            raise SystemExit(f"[verify] pytest collect failed rc={r.returncode}")
        print(f"[verify] OK: sympy imports at {commit[:8]}, collect rc={r.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["fetch", "clone", "venv", "verify", "all"])
    ap.add_argument("--root", default="/workspace/swebench")
    ap.add_argument("--repo", default="sympy/sympy")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--min-version", default="1.11",
                    help="drop instances older than this (old sympy needs py<3.10)")
    ap.add_argument("--venv-tag", default="py311")
    ap.add_argument("--deps", nargs="*", default=["pytest", "mpmath"])
    args = ap.parse_args()
    if args.step in ("fetch", "all"):
        step_fetch(args.root, args.repo, args.limit, args.min_version)
    if args.step in ("clone", "all"):
        step_clone(args.root, args.repo)
    if args.step in ("venv", "all"):
        step_venv(args.root, args.venv_tag, args.deps)
    if args.step in ("verify", "all"):
        step_verify(args.root, args.repo, args.venv_tag)


if __name__ == "__main__":
    main()
