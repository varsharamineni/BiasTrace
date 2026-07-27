#!/usr/bin/env python
"""Stage 2/3: Score saved BBQ best-of-N candidates with one or more judges.

Scoring ONLY — no selection, no metrics. Takes the bbq_*_results.json files
from generate_bbq_bon.py and writes each judge's verdict into every candidate
under `judge_scores[<judge_name>]` = {score, raw_score, reply, passed}.
Selection and the accuracy/bias comparison happen afterwards in
compare_bbq_methods.py, so judging can be re-run, extended with new judges,
or repeated after regeneration without touching the other stages.

The generation-time judge no longer exists — the "original" judge is just a
normal spec here (call it BiasTrace in judges.json). Files produced by the
old joint script are still supported: legacy per-candidate score/passed
fields are mirrored into judge_scores["BiasTrace"] automatically unless a
judge of that name is being (re)scored.

Judge specs (inline --judge, repeatable, and/or --judges_config JSON list):
    name (required), type ("llm" default | "prm"),
    prompt (required for llm), score_field (default "score"),
    invert (default false), pass_score (required — stored in each file's
    metadata so stage 3 knows the threshold),
    judge_on ("reasoning"|"full", default "reasoning"),
    model / temperature / top_p / max_tokens (llm),
    aggregate (mean|min|max|last) / module / class / path (prm).

PRM judges score locally and sequentially (question format
"{context} {question}", exactly like FRM_baseline/run_eval.py); LLM judges
run concurrently. Resumable: already-scored candidates are skipped unless
--overwrite.

Judge credentials (LLM judges only):
    export LLM_BASE_URL="https://api.deepseek.com/v1"
    export LLM_API_KEY="..."

Example:
    python judge_bbq_candidates.py --input outputs/bbq_bon_run \
        --judges_config bon_new/judges.json
"""
import argparse
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from openai import OpenAI


LEGACY_JUDGE_NAME = "BiasTrace"   # name given to old joint-script scores

FILE_RE = re.compile(r"^bbq_(.+?)_results\.json$")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Score saved BBQ best-of-N candidates with LLM/PRM judges "
                    "(scoring only; run compare_bbq_methods.py afterwards)"
    )
    p.add_argument("--input", type=str, required=True,
                   help="A bbq_*_results.json file OR a directory containing them")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write annotated files (default: annotate in "
                        "place, keeping a .bak of each file)")
    p.add_argument("--judge", action="append", default=[],
                   help="Inline judge spec 'name=...,prompt=...,score_field=...,"
                        "invert=true,pass_score=0[,judge_on=full][,model=...]'. "
                        "Repeatable.")
    p.add_argument("--judges_config", type=str, default=None,
                   help="JSON file with a list of judge spec objects (same keys)")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-score candidates even if they already carry a score "
                        "for a judge name")
    # Global judge client defaults (per-judge specs can override most of these)
    p.add_argument("--judge_model", type=str, default="deepseek-v4-flash")
    p.add_argument("--judge_temperature", type=float, default=1.0)
    p.add_argument("--judge_top_p", type=float, default=0.9)
    p.add_argument("--judge_max_tokens", type=int, default=1024)
    p.add_argument("--judge_max_workers", type=int, default=8)
    p.add_argument("--judge_max_retries", type=int, default=3)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Judge specs (identical to the COMPAS stage-2 script)
# --------------------------------------------------------------------------- #
@dataclass
class JudgeSpec:
    name: str
    type: str = "llm"                    # "llm" | "prm"
    prompt_path: str = ""
    prompt_template: str = ""
    score_field: str = "score"
    invert: bool = False
    pass_score: float = 0.0
    judge_on: str = "reasoning"          # "reasoning" | "full"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 1024
    # prm-only
    aggregate: str = "mean"              # mean | min | max | last
    sigmoid: bool = False                # apply σ to each step score before
                                         # aggregating (use when score_trace
                                         # returns raw logits, so that
                                         # r_k = mean_t σ(f_θ(z_k,t)))
    prm_module: str = "bias_detection"
    prm_class: str = "FairnessPRM"
    prm_path: str = "reasoning_eval/FRM_baseline"


def resolve_prompt_path(name_or_path: str) -> str:
    candidates = [
        name_or_path,
        f"{name_or_path}.txt",
        os.path.join("reasoning_eval", "prompts", f"{name_or_path}.txt"),
        os.path.join("reasoning_eval", f"{name_or_path}.txt"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Judge prompt '{name_or_path}' not found. Tried: {candidates}")


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def parse_judge_specs(args) -> List[JudgeSpec]:
    raw: List[Dict[str, Any]] = []
    if args.judges_config:
        with open(args.judges_config) as f:
            cfg = json.load(f)
        if not isinstance(cfg, list):
            raise SystemExit("--judges_config must contain a JSON list of judge objects")
        raw.extend(cfg)
    for spec_str in args.judge:
        d: Dict[str, Any] = {}
        for part in spec_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"Bad --judge fragment '{part}' in '{spec_str}' "
                                 f"(expected key=value)")
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
        raw.append(d)

    if not raw:
        raise SystemExit("No judges given. Use --judge and/or --judges_config.")

    specs: List[JudgeSpec] = []
    seen = set()
    for d in raw:
        jtype = str(d.get("type", "llm")).lower()
        if jtype not in ("llm", "prm"):
            raise SystemExit(f"Judge type must be 'llm' or 'prm': {d}")
        if "name" not in d:
            raise SystemExit(f"Judge spec missing 'name': {d}")
        if jtype == "llm" and "prompt" not in d:
            raise SystemExit(f"LLM judge spec missing 'prompt': {d}")
        if "pass_score" not in d:
            raise SystemExit(f"Judge spec '{d['name']}' missing 'pass_score' — it "
                             f"defines which candidates this judge passes and is "
                             f"stored for the comparison stage")
        name = str(d["name"])
        if name in seen:
            raise SystemExit(f"Duplicate judge name '{name}'")
        seen.add(name)

        if jtype == "llm":
            prompt_path = resolve_prompt_path(str(d["prompt"]))
            with open(prompt_path) as f:
                template = f.read()
        else:
            prompt_path, template = "", ""

        spec = JudgeSpec(
            name=name,
            type=jtype,
            prompt_path=prompt_path,
            prompt_template=template,
            score_field=str(d.get("score_field", "score")),
            invert=_parse_bool(d.get("invert", False)),
            pass_score=float(d["pass_score"]),
            judge_on=str(d.get("judge_on", "reasoning")),
            model=str(d.get("model", args.judge_model)),
            temperature=float(d.get("temperature", args.judge_temperature)),
            top_p=float(d.get("top_p", args.judge_top_p)),
            max_tokens=int(d.get("max_tokens", args.judge_max_tokens)),
            aggregate=str(d.get("aggregate", "mean")),
            sigmoid=_parse_bool(d.get("sigmoid", False)),
            prm_module=str(d.get("module", "bias_detection")),
            prm_class=str(d.get("class", "FairnessPRM")),
            prm_path=str(d.get("path", "reasoning_eval/FRM_baseline")),
        )
        if spec.judge_on not in ("reasoning", "full"):
            raise SystemExit(f"judge_on must be 'reasoning' or 'full' "
                             f"(judge '{name}' has '{spec.judge_on}')")
        if spec.aggregate not in ("mean", "min", "max", "last"):
            raise SystemExit(f"aggregate must be mean|min|max|last "
                             f"(judge '{name}' has '{spec.aggregate}')")
        specs.append(spec)
    return specs


# --------------------------------------------------------------------------- #
# Scorers (identical behaviour to the joint script / COMPAS stage 2)
# --------------------------------------------------------------------------- #
class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class JudgeClient:
    def __init__(self, max_workers: int, max_retries: int):
        base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise EnvironmentError("LLM_API_KEY is not set (export LLM_API_KEY=...)")
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.n_calls = 0

    def _parse_score(self, text: str, score_field: str) -> Optional[float]:
        cleaned = re.sub(r"```(?:json)?", "", text)
        for m in re.finditer(r"\{[^{}]*\}", cleaned, flags=re.DOTALL):
            try:
                obj = json.loads(m.group(0))
                if score_field in obj:
                    return float(obj[score_field])
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        m = re.search(rf"{re.escape(score_field)}\s*[\"']?\s*[*:=\s]*\**\s*(-?\d+(?:\.\d+)?)",
                      text, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
        return None

    def score_one(self, spec: JudgeSpec, fill: dict) -> Dict[str, Any]:
        prompt = spec.prompt_template.format_map(SafeDict(fill))
        for attempt in range(1, self.max_retries + 1):
            try:
                self.n_calls += 1
                resp = self.client.chat.completions.create(
                    model=spec.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=spec.temperature,
                    top_p=spec.top_p,
                    max_tokens=spec.max_tokens,
                )
                reply = resp.choices[0].message.content or ""
                raw = self._parse_score(reply, spec.score_field)
                if raw is not None:
                    score = (-raw if spec.invert else raw) + 0.0
                    return {"score": score, "raw_score": raw, "reply": reply,
                            "passed": score >= spec.pass_score}
                print(f"WARNING: [{spec.name}] unparseable reply for field "
                      f"'{spec.score_field}' (attempt {attempt}): {reply[:150]!r}")
            except Exception as e:
                print(f"WARNING: [{spec.name}] judge call failed (attempt {attempt}): {e}")
                time.sleep(2 * attempt)
        return {"score": None, "raw_score": None, "reply": "", "passed": False}


class PRMScorer:
    def __init__(self):
        self._instances: Dict[Tuple[str, str], Any] = {}
        self.n_scored = 0

    def _get(self, spec: JudgeSpec):
        key = (spec.prm_module, spec.prm_class)
        if key not in self._instances:
            import importlib
            import sys
            if spec.prm_path and os.path.isdir(spec.prm_path) \
                    and spec.prm_path not in sys.path:
                sys.path.insert(0, spec.prm_path)
            try:
                mod = importlib.import_module(spec.prm_module)
            except ImportError as e:
                raise SystemExit(
                    f"Cannot import PRM module '{spec.prm_module}' for judge "
                    f"'{spec.name}' (tried sys.path incl. '{spec.prm_path}'): {e}"
                )
            print(f"Loading PRM {spec.prm_module}.{spec.prm_class} ...")
            self._instances[key] = getattr(mod, spec.prm_class)()
        return self._instances[key]

    @staticmethod
    def _aggregate(step_scores: List[float], how: str) -> float:
        if not step_scores:
            return 0.5                       # same default as run_eval.py
        if how == "mean":
            return sum(step_scores) / len(step_scores)
        if how == "min":
            return min(step_scores)
        if how == "max":
            return max(step_scores)
        return step_scores[-1]               # "last"

    def score_one(self, spec: JudgeSpec, question: str, trace: str) -> Dict[str, Any]:
        prm = self._get(spec)
        try:
            step_scores, _ = prm.score_trace(question, trace)
        except Exception as e:
            print(f"WARNING: [{spec.name}] PRM scoring failed: {e}")
            return {"score": None, "raw_score": None, "reply": "",
                    "passed": False, "step_scores": []}
        self.n_scored += 1
        step_scores = [float(s) for s in (step_scores or [])]
        if spec.sigmoid:
            import math
            step_scores = [1.0 / (1.0 + math.exp(-s)) for s in step_scores]
        raw = self._aggregate(step_scores, spec.aggregate)
        score = (-raw if spec.invert else raw) + 0.0
        return {"score": score, "raw_score": raw,
                "reply": ", ".join(f"{s:.3f}" for s in step_scores),
                "passed": score >= spec.pass_score,
                "step_scores": step_scores, "aggregate": spec.aggregate,
                "sigmoid": spec.sigmoid}


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def find_result_files(input_path: str) -> List[Tuple[str, str]]:
    """Returns [(category, path)] for bbq_*_results.json files."""
    if os.path.isfile(input_path):
        m = FILE_RE.match(os.path.basename(input_path))
        return [(m.group(1) if m else "unknown", input_path)]
    if not os.path.isdir(input_path):
        raise SystemExit(f"--input path does not exist: {input_path}")
    out = []
    for f in sorted(os.listdir(input_path)):
        m = FILE_RE.match(f)
        if m and m.group(1) != "all_categories":
            out.append((m.group(1), os.path.join(input_path, f)))
    if not out:
        raise SystemExit(f"No bbq_*_results.json files found in {input_path}")
    return out


def build_fill(row: Dict[str, Any], cand: Dict[str, Any], judge_on: str) -> dict:
    options_str = "\n".join(
        f"{chr(65 + i)}. {o}" for i, o in enumerate(row["answer_options"])
    )
    judged_text = cand.get("reasoning") if judge_on == "reasoning" else cand.get("text")
    judged_text = judged_text or cand.get("text", "")
    return {
        "context": row.get("context", ""),
        "question": row.get("question", ""),
        "options": options_str,
        "answer": judged_text,
        "reasoning_trace": judged_text,
        "response": judged_text,
        "model_output": cand.get("text", ""),
        "model_reasoning": cand.get("reasoning", ""),
        "model_answer": cand.get("normalized_answer", ""),
        **{f"ans{i}": o for i, o in enumerate(row["answer_options"])},
    }


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    specs = parse_judge_specs(args)
    client = (JudgeClient(args.judge_max_workers, args.judge_max_retries)
              if any(s.type == "llm" for s in specs) else None)
    prm = PRMScorer()

    files = find_result_files(args.input)
    print(f"Result files: {len(files)}")
    for cat, f in files:
        print(f"  - [{cat}] {f}")
    print(f"Judges to run: {', '.join(f'{s.name}({s.type})' for s in specs)}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    spec_names = {s.name for s in specs}
    grand_total = 0
    grand_counts: Dict[str, Dict[str, int]] = {}

    for category, path in files:
        with open(path) as f:
            payload = json.load(f)
        rows = payload["results"] if isinstance(payload, dict) else payload

        missing = [i for i, r in enumerate(rows) if not r.get("candidates")]
        if missing:
            raise SystemExit(
                f"{path}: {len(missing)} rows have no 'candidates' — regenerate "
                f"with generate_bbq_bon.py (which always stores them)."
            )

        # Backward compat: mirror old joint-script generation-time scores
        if LEGACY_JUDGE_NAME not in spec_names:
            mirrored = 0
            for r in rows:
                for cand in r["candidates"]:
                    if cand.get("score") is None and not cand.get("judge_reply"):
                        continue
                    js = cand.setdefault("judge_scores", {})
                    if LEGACY_JUDGE_NAME not in js:
                        js[LEGACY_JUDGE_NAME] = {
                            "score": cand.get("score"),
                            "raw_score": None,
                            "reply": cand.get("judge_reply", ""),
                            "passed": bool(cand.get("passed", False)),
                        }
                        mirrored += 1
            if mirrored:
                print(f"  [{category}] mirrored legacy generation-time scores "
                      f"into judge_scores['{LEGACY_JUDGE_NAME}'] "
                      f"for {mirrored} candidates")

        # ---- worklist ------------------------------------------------------- #
        work = []
        for r in rows:
            for c in r["candidates"]:
                for spec in specs:
                    existing = c.get("judge_scores", {}).get(spec.name, {})
                    if existing.get("score") is not None and not args.overwrite:
                        continue
                    work.append((r, c, spec))

        label = os.path.basename(path)
        llm_work = [(r, c, s) for r, c, s in work if s.type == "llm"]
        prm_work = [(r, c, s) for r, c, s in work if s.type == "prm"]

        if llm_work:
            with ThreadPoolExecutor(max_workers=args.judge_max_workers) as pool, \
                 tqdm(total=len(llm_work), desc=f"  {label} [llm]", unit="calls") as pbar:
                futures = {
                    pool.submit(client.score_one, spec, build_fill(r, c, spec.judge_on)):
                    (c, spec) for r, c, spec in llm_work
                }
                for fut in as_completed(futures):
                    c, spec = futures[fut]
                    c.setdefault("judge_scores", {})[spec.name] = fut.result()
                    pbar.update(1)
        if prm_work:
            for r, c, spec in tqdm(prm_work, desc=f"  {label} [prm]", unit="traces"):
                question = f"{r.get('context', '')} {r.get('question', '')}".strip()
                trace = (c.get("reasoning") if spec.judge_on == "reasoning"
                         else c.get("text")) or c.get("text", "")
                c.setdefault("judge_scores", {})[spec.name] = \
                    prm.score_one(spec, question, trace)
        if not work:
            print(f"  {label}: all candidates already scored for these judges "
                  f"(use --overwrite to redo)")

        # ---- judge registry in metadata (stage 3 reads pass_score here) ----- #
        if isinstance(payload, dict):
            registry = payload.setdefault("metadata", {}).setdefault("judges", {})
            for s in specs:
                registry[s.name] = {
                    "type": s.type,
                    "prompt": s.prompt_path or None,
                    "score_field": s.score_field if s.type == "llm" else None,
                    "invert": s.invert,
                    "pass_score": s.pass_score,
                    "judge_on": s.judge_on,
                    "model": (s.model if s.type == "llm"
                              else f"{s.prm_module}.{s.prm_class}"),
                    "aggregate": s.aggregate if s.type == "prm" else None,
                    "sigmoid": s.sigmoid if s.type == "prm" else None,
                }
            registry.setdefault(LEGACY_JUDGE_NAME,
                                {"type": "legacy", "pass_score": 0.0})

        # ---- write ----------------------------------------------------------- #
        if args.output_dir:
            out_path = os.path.join(args.output_dir, os.path.basename(path))
        else:
            out_path = path
            bak = path + ".bak"
            if not os.path.exists(bak):
                shutil.copy2(path, bak)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)

        # per-file diagnostics into the grand tally
        for r in rows:
            for c in r["candidates"]:
                grand_total += 1
                for name, entry in c.get("judge_scores", {}).items():
                    g = grand_counts.setdefault(name, {"scored": 0, "passed": 0})
                    if entry.get("score") is not None:
                        g["scored"] += 1
                    if entry.get("passed"):
                        g["passed"] += 1

    print(f"\nScoring summary ({grand_total} candidates across {len(files)} files):")
    for name in sorted(grand_counts):
        g = grand_counts[name]
        print(f"  {name:<24} scored {g['scored']}/{grand_total} "
              f"({100.0 * g['scored'] / grand_total:.1f}%) | passed {g['passed']} "
              f"({100.0 * g['passed'] / grand_total:.1f}%)")

    print(f"\nJudge API calls this run: {client.n_calls if client else 0}"
          + (f" | PRM traces scored: {prm.n_scored}" if prm.n_scored else ""))
    print(f"Next: compare_bbq_methods.py --input "
          f"{args.output_dir or args.input}")


if __name__ == "__main__":
    main()