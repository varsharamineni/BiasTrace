#!/usr/bin/env python
"""Stage 2/3: Score saved Bias in Bios best-of-N candidates with one or more judges.

Scoring ONLY — no selection, no metrics. Writes each judge's verdict into
every candidate under judge_scores[<name>] = {score, raw_score, reply, passed}.
Selection and the accuracy/fairness comparison happen in compare_bios_methods.py.

Identical judge-spec machinery to judge_compas_candidates.py; judge prompts
can reuse the same templates (fill {context}/{question}/{answer}/{reasoning_trace}
— for Bios, context is the biography, question is "What is this person's profession?").

Judge credentials:
    export LLM_BASE_URL="https://api.deepseek.com/v1"
    export LLM_API_KEY="..."

Example:
    python judge_bios_candidates.py --input outputs/bios_bon_run \
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

LEGACY_JUDGE_NAME = "BiasTrace"
ANSWER_OPTIONS = ["nurse", "physician", "teacher", "professor"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Score Bias in Bios best-of-N candidates with LLM/PRM judges"
    )
    p.add_argument("--input", type=str, required=True,
                   help="bios_results.json or a directory containing one")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write annotated file (default: in place, .bak kept)")
    p.add_argument("--judge", action="append", default=[])
    p.add_argument("--judges_config", type=str, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--judge_model", type=str, default="deepseek-v4-flash")
    p.add_argument("--judge_temperature", type=float, default=0.0)
    p.add_argument("--judge_top_p", type=float, default=0.9)
    p.add_argument("--judge_max_tokens", type=int, default=1024)
    p.add_argument("--judge_max_workers", type=int, default=8)
    p.add_argument("--judge_max_retries", type=int, default=3)
    return p.parse_args()


@dataclass
class JudgeSpec:
    name: str
    type: str = "llm"
    prompt_path: str = ""
    prompt_template: str = ""
    score_field: str = "score"
    invert: bool = False
    pass_score: float = 0.0
    judge_on: str = "reasoning"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 1024
    aggregate: str = "mean"
    prm_module: str = "bias_detection"
    prm_class: str = "FairnessPRM"
    prm_path: str = "reasoning_eval/FRM_baseline"
    sigmoid: bool = False


def resolve_prompt_path(name_or_path: str) -> str:
    for c in [name_or_path, f"{name_or_path}.txt",
              os.path.join("reasoning_eval", "prompts", f"{name_or_path}.txt"),
              os.path.join("reasoning_eval", f"{name_or_path}.txt")]:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Judge prompt '{name_or_path}' not found.")


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
            raise SystemExit("--judges_config must be a JSON list")
        raw.extend(cfg)
    for spec_str in args.judge:
        d: Dict[str, Any] = {}
        for part in spec_str.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"Bad --judge fragment '{part}'")
            k, v = part.split("=", 1)
            d[k.strip()] = v.strip()
        raw.append(d)
    if not raw:
        raise SystemExit("No judges given. Use --judge and/or --judges_config.")

    specs: List[JudgeSpec] = []
    seen = set()
    for d in raw:
        jtype = str(d.get("type", "llm")).lower()
        if "name" not in d:
            raise SystemExit(f"Judge spec missing 'name': {d}")
        if jtype == "llm" and "prompt" not in d:
            raise SystemExit(f"LLM judge missing 'prompt': {d}")
        if "pass_score" not in d:
            raise SystemExit(f"Judge '{d['name']}' missing 'pass_score'")
        name = str(d["name"])
        if name in seen:
            raise SystemExit(f"Duplicate judge name '{name}'")
        seen.add(name)
        prompt_path = template = ""
        if jtype == "llm":
            prompt_path = resolve_prompt_path(str(d["prompt"]))
            with open(prompt_path) as f:
                template = f.read()
        spec = JudgeSpec(
            name=name, type=jtype,
            prompt_path=prompt_path, prompt_template=template,
            score_field=str(d.get("score_field", "score")),
            invert=_parse_bool(d.get("invert", False)),
            pass_score=float(d["pass_score"]),
            judge_on=str(d.get("judge_on", "reasoning")),
            model=str(d.get("model", args.judge_model)),
            temperature=float(d.get("temperature", args.judge_temperature)),
            top_p=float(d.get("top_p", args.judge_top_p)),
            max_tokens=int(d.get("max_tokens", args.judge_max_tokens)),
            aggregate=str(d.get("aggregate", "mean")),
            prm_module=str(d.get("module", "bias_detection")),
            prm_class=str(d.get("class", "FairnessPRM")),
            prm_path=str(d.get("path", "reasoning_eval/FRM_baseline")),
            sigmoid=_parse_bool(d.get("sigmoid", False)),
        )
        specs.append(spec)
    return specs


class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


class JudgeClient:
    def __init__(self, max_workers: int, max_retries: int):
        base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1")
        api_key = os.environ.get("LLM_API_KEY")
        if not api_key:
            raise EnvironmentError("LLM_API_KEY is not set")
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
        m = re.search(
            rf"{re.escape(score_field)}\s*[\"']?\s*[*:=\s]*\**\s*(-?\d+(?:\.\d+)?)",
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
                print(f"WARNING: [{spec.name}] unparseable (attempt {attempt}): "
                      f"{reply[:150]!r}")
            except Exception as e:
                print(f"WARNING: [{spec.name}] call failed (attempt {attempt}): {e}")
                time.sleep(2 * attempt)
        return {"score": None, "raw_score": None, "reply": "", "passed": False}


class PRMScorer:
    def __init__(self):
        self._instances: Dict[Tuple[str, str], Any] = {}
        self.n_scored = 0

    def _get(self, spec: JudgeSpec):
        key = (spec.prm_module, spec.prm_class)
        if key not in self._instances:
            import importlib, sys
            if spec.prm_path and os.path.isdir(spec.prm_path) \
                    and spec.prm_path not in sys.path:
                sys.path.insert(0, spec.prm_path)
            try:
                mod = importlib.import_module(spec.prm_module)
            except ImportError as e:
                raise SystemExit(f"Cannot import PRM '{spec.prm_module}': {e}")
            print(f"Loading PRM {spec.prm_module}.{spec.prm_class} ...")
            self._instances[key] = getattr(mod, spec.prm_class)()
        return self._instances[key]

    @staticmethod
    def _aggregate(step_scores: List[float], how: str) -> float:
        if not step_scores:
            return 0.5
        if how == "mean":
            return sum(step_scores) / len(step_scores)
        if how == "min":
            return min(step_scores)
        if how == "max":
            return max(step_scores)
        return step_scores[-1]

    def score_one(self, spec: JudgeSpec, question: str, trace: str) -> Dict[str, Any]:
        import math
        prm = self._get(spec)
        try:
            step_scores, _ = prm.score_trace(question, trace)
        except Exception as e:
            print(f"WARNING: [{spec.name}] PRM failed: {e}")
            return {"score": None, "raw_score": None, "reply": "",
                    "passed": False, "step_scores": []}
        self.n_scored += 1
        step_scores = [float(s) for s in (step_scores or [])]
        if spec.sigmoid:
            step_scores = [1.0 / (1.0 + math.exp(-s)) for s in step_scores]
        raw = self._aggregate(step_scores, spec.aggregate)
        score = (-raw if spec.invert else raw) + 0.0
        return {"score": score, "raw_score": raw,
                "reply": ", ".join(f"{s:.3f}" for s in step_scores),
                "passed": score >= spec.pass_score,
                "step_scores": step_scores, "aggregate": spec.aggregate,
                "sigmoid": spec.sigmoid}


def find_result_file(input_path: str) -> str:
    if os.path.isfile(input_path):
        return input_path
    if not os.path.isdir(input_path):
        raise SystemExit(f"--input path does not exist: {input_path}")
    for name in ("bios_results.json",):
        p = os.path.join(input_path, name)
        if os.path.isfile(p):
            return p
    raise SystemExit(f"No bios_results.json found in {input_path}")


def build_fill(row: Dict[str, Any], cand: Dict[str, Any], judge_on: str) -> dict:
    """Build the template fill dict. For Bios:
       context = biography text, question = profession prediction task."""
    judged_text = (cand.get("reasoning") if judge_on == "reasoning"
                   else cand.get("text")) or cand.get("text", "")
    options_str = "\n".join(
        f"{chr(65+i)}. {o}" for i, o in enumerate(row["answer_options"])
    )
    return {
        "context": row.get("bio_text", ""),
        "question": "What is this person's profession?",
        "options": options_str,
        "answer": judged_text,
        "reasoning_trace": judged_text,
        "response": judged_text,
        "model_output": cand.get("text", ""),
        "model_reasoning": cand.get("reasoning", ""),
        "model_answer": cand.get("normalized_answer", ""),
        **{f"ans{i}": o for i, o in enumerate(row["answer_options"])},
    }


def main():
    args = parse_args()
    specs = parse_judge_specs(args)
    client = (JudgeClient(args.judge_max_workers, args.judge_max_retries)
              if any(s.type == "llm" for s in specs) else None)
    prm = PRMScorer()

    path = find_result_file(args.input)
    print(f"Result file: {path}")
    print(f"Judges to run: {', '.join(f'{s.name}({s.type})' for s in specs)}")

    with open(path) as f:
        payload = json.load(f)
    rows = payload["results"] if isinstance(payload, dict) else payload

    missing = [i for i, r in enumerate(rows) if not r.get("candidates")]
    if missing:
        raise SystemExit(
            f"{path}: {len(missing)} rows have no 'candidates' — regenerate with "
            f"generate_bios_bon.py or generate_bios_bon_api.py.")

    spec_names = {s.name for s in specs}
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
            print(f"Mirrored {mirrored} legacy scores into "
                  f"judge_scores['{LEGACY_JUDGE_NAME}']")

    work = [(r, c, s)
            for r in rows for c in r["candidates"] for s in specs
            if c.get("judge_scores", {}).get(s.name, {}).get("score") is None
            or args.overwrite]

    llm_work = [(r, c, s) for r, c, s in work if s.type == "llm"]
    prm_work = [(r, c, s) for r, c, s in work if s.type == "prm"]

    if llm_work:
        with ThreadPoolExecutor(max_workers=args.judge_max_workers) as pool, \
             tqdm(total=len(llm_work), desc="  [llm]", unit="calls") as pbar:
            futures = {
                pool.submit(client.score_one, spec,
                            build_fill(r, c, spec.judge_on)): (c, spec)
                for r, c, spec in llm_work
            }
            for fut in as_completed(futures):
                c, spec = futures[fut]
                c.setdefault("judge_scores", {})[spec.name] = fut.result()
                pbar.update(1)
    if prm_work:
        for r, c, spec in tqdm(prm_work, desc="  [prm]", unit="traces"):
            question = (f"Biography: {r.get('bio_text', '')} "
                        f"What is this person's profession?").strip()
            trace = (c.get("reasoning") if spec.judge_on == "reasoning"
                     else c.get("text")) or c.get("text", "")
            c.setdefault("judge_scores", {})[spec.name] = \
                prm.score_one(spec, question, trace)
    if not work:
        print("All candidates already scored (use --overwrite to redo)")

    if isinstance(payload, dict):
        registry = payload.setdefault("metadata", {}).setdefault("judges", {})
        for s in specs:
            registry[s.name] = {
                "type": s.type,
                "prompt": s.prompt_path or None,
                "score_field": s.score_field if s.type == "llm" else None,
                "invert": s.invert, "pass_score": s.pass_score,
                "judge_on": s.judge_on,
                "model": (s.model if s.type == "llm"
                          else f"{s.prm_module}.{s.prm_class}"),
                "aggregate": s.aggregate if s.type == "prm" else None,
                "sigmoid": s.sigmoid if s.type == "prm" else None,
            }
        registry.setdefault(LEGACY_JUDGE_NAME, {"type": "legacy", "pass_score": 0.0})

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
    else:
        out_path = path
        bak = path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    total = sum(len(r["candidates"]) for r in rows)
    print(f"\nScoring summary ({total} candidates):")
    all_names = sorted({n for r in rows for c in r["candidates"]
                        for n in c.get("judge_scores", {})})
    for name in all_names:
        scored = sum(1 for r in rows for c in r["candidates"]
                     if c.get("judge_scores", {}).get(name, {}).get("score") is not None)
        passed = sum(1 for r in rows for c in r["candidates"]
                     if c.get("judge_scores", {}).get(name, {}).get("passed"))
        print(f"  {name:<24} scored {scored}/{total} "
              f"({100.0*scored/total:.1f}%) | passed {passed} "
              f"({100.0*passed/total:.1f}%)")

    print(f"\nJudge API calls: {client.n_calls if client else 0}"
          + (f" | PRM traces: {prm.n_scored}" if prm.n_scored else ""))
    print(f"Annotated file: {out_path}")
    print(f"Next: compare_bios_methods.py --input {out_path}")


if __name__ == "__main__":
    main()