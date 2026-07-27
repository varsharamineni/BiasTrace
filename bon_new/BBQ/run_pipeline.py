#!/usr/bin/env python
"""One-command pipeline for the BBQ best-of-N experiments (3-stage layout).

Chains the three stage scripts, per run (model/dataset/seed combo):

    1. gen       generate_bbq_bon.py        (vLLM, NO judge, candidates saved)
    2. judge     judge_bbq_candidates.py    (all judges incl. BiasTrace; resumable)
    3. compare   compare_bbq_methods.py     (selection + accuracy/bias comparison)

The generation-time judge is gone — put the original judge in judges.json
like any other judge (e.g. name it BiasTrace). Everything is driven by one
JSON config; each run gets its own folder under output_root and completed
stages are skipped automatically, so re-running after a crash or after
adding a judge only does the missing work. The judge stage is itself
resumable, so it always runs (a no-op when nothing is new). At the end,
every run's judge_comparison.csv is concatenated into
output_root/all_runs_comparison.csv.

Quick start:

    python run_bbq_pipeline.py --init pipeline.json
    # edit pipeline.json (models, categories, judges_config, script paths)
    python run_bbq_pipeline.py --config pipeline.json

Useful flags:

    --dry_run              print the commands without running anything
    --stages gen,judge     run only some stages (gen | judge | compare | aggregate)
    --runs qwen3-8b        only these run names (repeatable / comma separated)
    --force                re-run stages even if their outputs look complete
    --keep_going           continue with the next run if one run fails
"""
import argparse
import copy
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from typing import Any, Dict, List

EXAMPLE_CONFIG = {
    "scripts": {
        "generate": "bon_new/generate_bbq_bon.py",
        "judge": "bon_new/judge_bbq_candidates.py",
        "compare": "bon_new/compare_bbq_methods.py",
    },
    "data_dir": "datasets/bbq_dataset_all_cat/data",
    "meta_file": "datasets/bbq_additional_metadata.csv",
    "categories": ["Age", "Nationality", "Religion"],
    "output_root": "outputs/pipeline",
    "generation": {
        "best_of_n": 8,
        "seed": 42,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "max_length": 2048,
        "batch_size": 32,
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "quiet": True,
    },
    "judges_config": "bon_new/judges.json",
    "judge_extra_args": [],
    "compare_extra_args": [],
    "runs": [
        {"name": "qwen3-1.7b_N8_s1", "model": "Qwen/Qwen3-1.7B",
         "generation": {"seed": 1}},
    ],
}

STAGES = ("gen", "judge", "compare", "aggregate")


# --------------------------------------------------------------------------- #
# Config handling
# --------------------------------------------------------------------------- #
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s).strip("_")


def resolve_runs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    global_keys = {k: v for k, v in cfg.items() if k != "runs"}
    runs = []
    seen = set()
    for i, run in enumerate(cfg.get("runs", [])):
        merged = deep_merge(global_keys, {k: v for k, v in run.items()
                                          if k != "name"})
        if "model" not in merged:
            raise SystemExit(f"Run #{i} has no 'model' (in run or globals)")
        gen = merged.get("generation", {})
        name = run.get("name") or (
            f"{sanitize(os.path.basename(str(merged['model']).rstrip('/')))}"
            f"_N{gen.get('best_of_n', 8)}_seed{gen.get('seed', 42)}"
        )
        if name in seen:
            raise SystemExit(f"Duplicate run name '{name}' — set explicit names")
        seen.add(name)
        merged["name"] = name
        merged["out_dir"] = os.path.join(merged.get("output_root",
                                                    "outputs/pipeline"), name)
        runs.append(merged)
    if not runs:
        raise SystemExit("Config has no runs")
    return runs


# --------------------------------------------------------------------------- #
# Command building
# --------------------------------------------------------------------------- #
def kwargs_to_cli(kwargs: Dict[str, Any]) -> List[str]:
    argv: List[str] = []
    extra = kwargs.get("extra_args", [])
    for k, v in kwargs.items():
        if k == "extra_args" or v is None or v is False:
            continue
        flag = f"--{k}"
        if v is True:
            argv.append(flag)
        elif isinstance(v, list):
            argv.append(flag)
            argv.extend(str(x) for x in v)
        else:
            argv.extend([flag, str(v)])
    argv.extend(str(x) for x in extra)
    return argv


def gen_cmd(run: Dict[str, Any]) -> List[str]:
    cmd = [sys.executable, run["scripts"]["generate"],
           "--model", str(run["model"]),
           "--output_dir", run["out_dir"],
           "--data_dir", run["data_dir"],
           "--categories", *run["categories"]]
    cmd += kwargs_to_cli(dict(run.get("generation", {})))
    return cmd


def judge_stage_cmd(run: Dict[str, Any]) -> List[str]:
    cmd = [sys.executable, run["scripts"]["judge"],
           "--input", run["out_dir"],
           "--judges_config", run["judges_config"]]
    cmd += [str(x) for x in run.get("judge_extra_args", [])]
    return cmd


def compare_cmd(run: Dict[str, Any]) -> List[str]:
    cmd = [sys.executable, run["scripts"]["compare"],
           "--input", run["out_dir"],
           "--data_dir", run["data_dir"],
           "--meta_file", run["meta_file"]]
    cmd += [str(x) for x in run.get("compare_extra_args", [])]
    return cmd


# --------------------------------------------------------------------------- #
# Stage completion checks
# --------------------------------------------------------------------------- #
def results_files(run) -> List[str]:
    return [os.path.join(run["out_dir"], f"bbq_{c}_results.json")
            for c in run["categories"]]


def newest(paths: List[str]) -> float:
    return max((os.path.getmtime(p) for p in paths if os.path.exists(p)),
               default=0.0)


def gen_done(run) -> bool:
    return all(os.path.exists(p) for p in results_files(run))


def compare_done(run) -> bool:
    stats = os.path.join(run["out_dir"], "judge_comparison_stats.json")
    if not os.path.exists(stats):
        return False
    # stale if the (judged) result files OR the judges config changed since
    inputs = results_files(run) + [run["judges_config"]]
    return os.path.getmtime(stats) >= newest(inputs)


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def run_stage(name: str, cmd: List[str], dry_run: bool, log_dir: str) -> None:
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"\n>>> [{name}] {printable}")
    if dry_run:
        return
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    t0 = time.time()
    with open(log_path, "a") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
                  f"{printable}\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:                     # tee: console + log file
            print(line, end="")
            log.write(line)
        proc.wait()
    mins = (time.time() - t0) / 60
    if proc.returncode != 0:
        raise RuntimeError(f"Stage '{name}' failed (exit {proc.returncode}, "
                           f"{mins:.1f} min) — full log: {log_path}")
    print(f"<<< [{name}] done in {mins:.1f} min (log: {log_path})")


def aggregate(runs: List[Dict[str, Any]], output_root: str,
              dry_run: bool) -> None:
    out_csv = os.path.join(output_root, "all_runs_comparison.csv")
    print(f"\n>>> [aggregate] combining per-run judge_comparison.csv "
          f"-> {out_csv}")
    if dry_run:
        return
    combined: List[Dict[str, Any]] = []
    for run in runs:
        path = os.path.join(run["out_dir"], "judge_comparison.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} missing (compare stage not run?) — skipped")
            continue
        gen = run.get("generation", {})
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                combined.append({
                    "run": run["name"],
                    "model": run["model"],
                    "seed": gen.get("seed", ""),
                    "best_of_n": gen.get("best_of_n", ""),
                    **row,
                })
    if not combined:
        print("  Nothing to aggregate.")
        return
    fieldnames = list(combined[0].keys())
    for row in combined[1:]:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    os.makedirs(output_root, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(combined)
    print(f"  {len(combined)} rows from "
          f"{len({r['run'] for r in combined})} runs -> {out_csv}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Run the BBQ best-of-N pipeline (generate -> judge -> "
                    "compare) for one or more models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=str, help="Pipeline JSON config")
    ap.add_argument("--init", type=str, metavar="PATH",
                    help="Write an example config to PATH and exit")
    ap.add_argument("--stages", type=str, default="gen,judge,compare,aggregate",
                    help=f"Comma-separated subset of {STAGES}")
    ap.add_argument("--runs", type=str, action="append", default=[],
                    help="Only run these run names (repeatable / comma-sep)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run stages even if their outputs look complete")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--keep_going", action="store_true",
                    help="If a run fails, continue with the next run")
    args = ap.parse_args()

    if args.init:
        if os.path.exists(args.init):
            raise SystemExit(f"{args.init} already exists — not overwriting")
        with open(args.init, "w") as f:
            json.dump(EXAMPLE_CONFIG, f, indent=2)
        print(f"Wrote example config to {args.init}. Edit it, then:\n"
              f"  python {sys.argv[0]} --config {args.init}")
        return
    if not args.config:
        ap.error("--config is required (or --init to create one)")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        raise SystemExit(f"Unknown stage(s) {bad}; valid: {STAGES}")

    with open(args.config) as f:
        cfg = json.load(f)
    runs = resolve_runs(cfg)

    wanted = {n.strip() for arg in args.runs for n in arg.split(",") if n.strip()}
    if wanted:
        unknown = wanted - {r["name"] for r in runs}
        if unknown:
            raise SystemExit(f"Unknown run name(s) {sorted(unknown)}; "
                             f"available: {[r['name'] for r in runs]}")
        runs = [r for r in runs if r["name"] in wanted]

    # fail fast on missing scripts / judges config before any GPU time
    for r in runs:
        for stage, key in (("gen", "generate"), ("judge", "judge"),
                           ("compare", "compare")):
            if stage in stages and not os.path.isfile(r["scripts"][key]):
                raise SystemExit(f"[{r['name']}] scripts.{key} not found: "
                                 f"{r['scripts'][key]}")
        if "judge" in stages and not os.path.isfile(r["judges_config"]):
            raise SystemExit(f"[{r['name']}] judges_config not found: "
                             f"{r['judges_config']}")

    print(f"Pipeline: {len(runs)} run(s), stages: {', '.join(stages)}"
          f"{'  [DRY RUN]' if args.dry_run else ''}")
    for r in runs:
        print(f"  - {r['name']}: model={r['model']}  ->  {r['out_dir']}")

    failed: List[str] = []
    for run in runs:
        print(f"\n{'#' * 70}\n# RUN: {run['name']}\n{'#' * 70}")
        log_dir = os.path.join(run["out_dir"], "logs")
        try:
            if "gen" in stages:
                if gen_done(run) and not args.force:
                    print(f">>> [gen] outputs already present in "
                          f"{run['out_dir']} — skipping (use --force to redo)")
                else:
                    run_stage("gen", gen_cmd(run), args.dry_run, log_dir)
            if "judge" in stages:
                # the judge stage is itself resumable: always run it — a no-op
                # when everything is already scored, and it picks up new
                # judges/candidates otherwise
                run_stage("judge", judge_stage_cmd(run), args.dry_run, log_dir)
            if "compare" in stages:
                if compare_done(run) and not args.force:
                    print(">>> [compare] comparison up to date — skipping")
                else:
                    run_stage("compare", compare_cmd(run), args.dry_run, log_dir)
        except (RuntimeError, KeyboardInterrupt) as e:
            if isinstance(e, KeyboardInterrupt):
                raise
            print(f"ERROR in run '{run['name']}': {e}")
            failed.append(run["name"])
            if not args.keep_going:
                raise SystemExit(1)

    if "aggregate" in stages:
        ok_runs = [r for r in runs if r["name"] not in failed]
        aggregate(ok_runs, cfg.get("output_root", "outputs/pipeline"),
                  args.dry_run)

    print(f"\n{'=' * 70}")
    if failed:
        print(f"Pipeline finished with failures: {failed}")
        sys.exit(1)
    print("Pipeline complete.")


if __name__ == "__main__":
    main()