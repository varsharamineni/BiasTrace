#!/usr/bin/env python
"""Drive the Bias in Bios best-of-N pipeline from a config JSON.

Three stages per run:
    1. generate  — generate_bios_bon.py (local vLLM) OR
                   generate_bios_bon_api.py (GPT-compatible API)
                   Set "use_api": true in the config to use the API script.
    2. judge     — judge_bios_candidates.py (resumable; always runs)
    3. compare   — compare_bios_methods.py (pure post-processing)

Usage:
    python run_pipeline_bios.py --config bon_new/BiasInBios/pipeline.json
    python run_pipeline_bios.py --config ... --dry_run
    python run_pipeline_bios.py --config ... --skip_generate
    python run_pipeline_bios.py --config ... --skip_generate --skip_judge
    python run_pipeline_bios.py --config ... --force
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

GEN_FLAG_KEYS = {"test_mode", "quiet", "enable_thinking"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true",
                    help="Re-run generation even if bios_results.json exists")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print commands only")
    ap.add_argument("--skip_generate", action="store_true")
    ap.add_argument("--skip_judge", action="store_true")
    ap.add_argument("--skip_compare", action="store_true")
    return ap.parse_args()


def build_generate_cmd(cfg: dict, run: dict, out_dir: str) -> list:
    gen = {**cfg.get("generation", {}), **run.get("generation", {})}
    use_api = cfg.get("use_api", False) or run.get("use_api", False)
    script = cfg["scripts"]["generate_api"] if use_api else cfg["scripts"]["generate"]
    cmd = [sys.executable, script,
           "--model", run["model"],
           "--output_dir", out_dir]
    if use_api and run.get("api_url"):
        cmd += ["--api_url", run["api_url"]]
    elif use_api and cfg.get("api_url"):
        cmd += ["--api_url", cfg["api_url"]]
    for key, val in gen.items():
        if val is None:
            continue
        if key in GEN_FLAG_KEYS:
            if val:
                cmd.append(f"--{key}")
        elif isinstance(val, bool):
            raise SystemExit(f"Boolean generation key '{key}' not in "
                             f"GEN_FLAG_KEYS — add it or pass a value")
        else:
            cmd += [f"--{key}", str(val)]
    return cmd


def run_cmd(cmd: list, dry: bool) -> None:
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}")
    if dry:
        return
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    for run in cfg["runs"]:
        name = run["name"]
        out_dir = os.path.join(cfg["output_root"], name)
        print(f"\n{'='*70}\nRUN: {name}  ->  {out_dir}\n{'='*70}")

        results_json = os.path.join(out_dir, "bios_results.json")

        if not args.skip_generate:
            if os.path.isfile(results_json) and not args.force:
                print(f"generation skipped: {results_json} exists "
                      f"(--force to redo)")
            else:
                run_cmd(build_generate_cmd(cfg, run, out_dir), args.dry_run)

        if not args.skip_judge:
            cmd = [sys.executable, cfg["scripts"]["judge"],
                   "--input", out_dir,
                   "--judges_config", cfg["judges_config"]]
            cmd += list(cfg.get("judge_extra_args", []))
            run_cmd(cmd, args.dry_run)

        if not args.skip_compare:
            cmd = [sys.executable, cfg["scripts"]["compare"],
                   "--input", out_dir]
            cmd += list(cfg.get("compare_extra_args", []))
            run_cmd(cmd, args.dry_run)

    print("\nPipeline complete." if not args.dry_run else "\nDry run complete.")


if __name__ == "__main__":
    main()