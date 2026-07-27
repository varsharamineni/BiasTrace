#!/usr/bin/env python
"""Drive the 3-stage COMPAS best-of-N pipeline from a config JSON.

Stages (each independently skippable / re-runnable):
    1. generate  - generate_compas_bon.py       (GPU; no judge, no API key)
    2. judge     - judge_compas_candidates.py   (scores candidates; resumable,
                   so re-running after adding judges only scores what's new)
    3. compare   - compare_compas_methods.py    (selection + accuracy/EO/EOpp;
                   instant, pure post-processing)

Generation is skipped when compas_results.json already exists (--force to
redo — note that regenerating invalidates existing judge scores, so the
judge stage will then re-score everything). The judge stage always runs
(its own resumability makes it a no-op when nothing is new). The compare
stage always runs; it is cheap.

The "generation" block holds the base flags; per-run "generation" blocks
override key-by-key. The original judge is NOT part of generation any more —
put it in judges.json like any other judge (e.g. name it BiasTrace).

Usage:
    python run_pipeline_compas.py --config bon_new/compas_pipeline.json
    python run_pipeline_compas.py --config ... --dry_run
    python run_pipeline_compas.py --config ... --force
    python run_pipeline_compas.py --config ... --skip_generate --skip_judge
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

GEN_FLAG_KEYS = {"test_mode", "quiet", "enable_thinking"}


def build_generate_cmd(cfg: dict, run: dict, out_dir: str) -> list:
    gen = {**cfg.get("generation", {}), **run.get("generation", {})}
    cmd = [sys.executable, cfg["scripts"]["generate"],
           "--model", run["model"],
           "--output_dir", out_dir]
    for key, val in gen.items():
        if val is None:
            continue
        if key in GEN_FLAG_KEYS:
            if val:
                cmd.append(f"--{key}")
        elif isinstance(val, bool):
            raise SystemExit(f"Boolean generation key '{key}' not in GEN_FLAG_KEYS — "
                             f"add it there or pass a value instead")
        else:
            cmd += [f"--{key}", str(val)]
    return cmd


def run_cmd(cmd: list, dry: bool) -> None:
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}")
    if dry:
        return
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true",
                    help="Re-run generation even if compas_results.json exists")
    ap.add_argument("--dry_run", action="store_true", help="Print commands only")
    ap.add_argument("--skip_generate", action="store_true")
    ap.add_argument("--skip_judge", action="store_true")
    ap.add_argument("--skip_compare", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    for run in cfg["runs"]:
        name = run["name"]
        out_dir = os.path.join(cfg["output_root"], name)
        print(f"\n{'=' * 70}\nRUN: {name}  ->  {out_dir}\n{'=' * 70}")

        results_json = os.path.join(out_dir, "compas_results.json")

        if not args.skip_generate:
            if os.path.isfile(results_json) and not args.force:
                print(f"generation skipped: {results_json} exists (--force to redo)")
            else:
                run_cmd(build_generate_cmd(cfg, run, out_dir), args.dry_run)

        if not args.skip_judge:
            cmd = [sys.executable, cfg["scripts"]["judge"], "--input", out_dir]
            if cfg.get("judges_config"):
                cmd += ["--judges_config", cfg["judges_config"]]
            cmd += list(cfg.get("judge_extra_args", []))
            run_cmd(cmd, args.dry_run)

        if not args.skip_compare:
            cmd = [sys.executable, cfg["scripts"]["compare"], "--input", out_dir]
            cmd += list(cfg.get("compare_extra_args", []))
            run_cmd(cmd, args.dry_run)

    print("\nPipeline complete." if not args.dry_run else "\nDry run complete.")


if __name__ == "__main__":
    main()