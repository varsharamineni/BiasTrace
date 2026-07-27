#!/usr/bin/env python
"""Stage 3/3: Selection + accuracy/fairness comparison for judged COMPAS
best-of-N outputs. Pure post-processing — no GPU, no API calls — so it can be
re-run instantly after regeneration, after adding judges, or with different
pass thresholds.

For every judge found in the candidates' judge_scores (or the subset given
via --judges) it recomputes the judge-filtered majority-vote answer, then
compares all methods on accuracy AND fairness:

    baselines:  first_sample, majority_all, oracle (pass@N)
    per judge:  filtered:<name>  (majority over judge-passed candidates,
                with --no_pass_fallback when nothing passed)

Fairness definitions match the COMPAS analysis pipeline (in %):
    EOpp gap = |TPR_A - TPR_B|
    EO gap   = |TPR_A - TPR_B| + |FPR_A - FPR_B|
    DP gap   = |high-risk-rate_A - high-risk-rate_B|
with y_pred = 1 iff the answer is "High risk" (Unknown counts as 0) and
y_true = recid_label.

Pass thresholds come from the judge registry that judge_compas_candidates.py
stores in metadata.judges; each candidate's stored `passed` flag is used
directly unless the threshold is overridden with (repeatable)
    --pass_score <judge_name>=<value>
in which case pass/fail is recomputed from the stored scores — so threshold
sweeps need no re-judging.

Outputs (next to the input file or in --output_dir):
    compas_results_compared.json   input + judge_selections per row +
                                   per-method pred/error enrichment
    judge_comparison_stats.json    full per-judge stats incl. per-race rates
    judge_comparison.csv           one row per method

Example:
    python compare_compas_methods.py --input outputs/compas_bon_run
    python compare_compas_methods.py --input outputs/compas_bon_run \
        --pass_score fairness_prm=0.7 --pass_score BiasTrace=0
"""
import argparse
import csv
import itertools
import json
import os
from typing import Any, Dict, List, Optional, Tuple

HIGH = "High risk"

BASELINES = {
    "first_sample": "first_sample_answer",
    "majority_all": "majority_answer",
}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Compare COMPAS BoN selection methods (baselines + one "
                    "filtered method per judge) on accuracy and EO/EOpp"
    )
    p.add_argument("--input", type=str, required=True,
                   help="A judged compas_results.json OR a directory containing one")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write outputs (default: next to the input file)")
    p.add_argument("--judges", nargs="+", default=None,
                   help="Only compare these judge names (default: every judge "
                        "found in the candidates)")
    p.add_argument("--pass_score", action="append", default=[],
                   help="Override a judge's pass threshold: name=value. "
                        "Repeatable. Pass/fail is recomputed from stored "
                        "scores, no re-judging needed.")
    p.add_argument("--no_pass_fallback", choices=["majority_all", "first_sample"],
                   default="majority_all",
                   help="Fallback when no candidate passes a judge")
    return p.parse_args()


def find_result_file(input_path: str) -> str:
    if os.path.isfile(input_path):
        return input_path
    if not os.path.isdir(input_path):
        raise SystemExit(f"--input path does not exist: {input_path}")
    for name in ("compas_results.json", "compas_results_corrected.json"):
        p = os.path.join(input_path, name)
        if os.path.isfile(p):
            return p
    raise SystemExit(f"No compas_results.json found in {input_path}")


# --------------------------------------------------------------------------- #
# Selection (identical semantics to the old joint script)
# --------------------------------------------------------------------------- #
def majority(cands: List[dict]) -> Tuple[str, int, int, Dict[str, int]]:
    counts: Dict[str, int] = {}
    first_seen: Dict[str, int] = {}
    for i, c in enumerate(cands):
        a = c.get("normalized_answer", "")
        if not a:
            continue
        counts[a] = counts.get(a, 0) + 1
        first_seen.setdefault(a, i)
    if not counts:
        return "", 0, 0, {}
    ans = min(counts, key=lambda a: (-counts[a], first_seen[a]))
    ranked = sorted(counts.values(), reverse=True)
    margin = counts[ans] - (ranked[1] if len(ranked) > 1 else 0)
    return ans, counts[ans], margin, counts


def candidate_passed(cand: dict, judge_name: str,
                     override: Optional[float]) -> Tuple[Optional[float], bool]:
    entry = cand.get("judge_scores", {}).get(judge_name, {})
    score = entry.get("score")
    if override is None:
        return score, bool(entry.get("passed"))
    return score, (score is not None and score >= override)


def select_for_judge(row: Dict[str, Any], judge_name: str,
                     no_pass_fallback: str,
                     override: Optional[float]) -> Dict[str, Any]:
    cands = row["candidates"]
    correct = row.get("correct_answer")

    scored = [candidate_passed(c, judge_name, override) for c in cands]
    passing = [c for c, (_, p) in zip(cands, scored) if p]
    n_judged = sum(1 for s, _ in scored if s is not None)

    maj_all, _, _, _ = majority(cands)
    maj_filt, votes_filt, margin_filt, counts_filt = majority(passing)

    fallback = ""
    if maj_filt:
        answer = maj_filt
    elif no_pass_fallback == "majority_all" and maj_all:
        answer, fallback = maj_all, "majority_all"
    else:
        answer, fallback = cands[0].get("normalized_answer", ""), "first_sample"

    return {
        "answer": answer,
        "pred": 1 if answer == HIGH else 0,   # Unknown counts as Low
        "is_correct": (bool(answer) and answer == correct) if correct else None,
        "num_passed": len(passing),
        "num_judged": n_judged,
        "num_unparseable": len(cands) - n_judged,
        "fallback_used": fallback,
        "votes": votes_filt,
        "margin": margin_filt,
        "answer_distribution": counts_filt,
        "changed_vs_majority_all": bool(answer) and answer != maj_all,
    }


# --------------------------------------------------------------------------- #
# Fairness stats (same definitions as the analysis pipeline, in %)
# --------------------------------------------------------------------------- #
def pct(num, den) -> Optional[float]:
    return 100.0 * num / den if den else None


def fairness_block(pairs: List[Tuple[Optional[str], Optional[int], int]]
                   ) -> Dict[str, Any]:
    """pairs: [(race, recid_label_or_None, pred01)]"""
    per: Dict[str, dict] = {}
    for race, label, pred in pairs:
        g = per.setdefault(race, {"Y1": 0, "Y0": 0, "TP": 0, "FP": 0,
                                  "n": 0, "pred1": 0})
        g["n"] += 1
        g["pred1"] += pred
        if label is None:
            continue
        if int(label) == 1:
            g["Y1"] += 1
            g["TP"] += pred
        else:
            g["Y0"] += 1
            g["FP"] += pred

    by_race = {
        race: {"n": g["n"],
               "tpr": pct(g["TP"], g["Y1"]),
               "fpr": pct(g["FP"], g["Y0"]),
               "high_risk_rate": pct(g["pred1"], g["n"])}
        for race, g in per.items()
    }

    def spread(key):
        vals = [v[key] for v in by_race.values() if v.get(key) is not None]
        return (max(vals) - min(vals)) if len(vals) >= 2 else None

    eopp, fpr_gap = spread("tpr"), spread("fpr")
    return {
        "by_race": by_race,
        "eopp_tpr_gap": eopp,
        "fpr_gap": fpr_gap,
        "eo_gap": (eopp + fpr_gap) if eopp is not None and fpr_gap is not None else None,
        "dp_gap": spread("high_risk_rate"),
    }


def accuracy_of(flags: List[Optional[bool]]) -> Optional[float]:
    labeled = [x for x in flags if x is not None]
    return pct(sum(labeled), len(labeled))


def summarize_judge(rows: List[Dict[str, Any]], judge_name: str, best_of_n: int
                    ) -> Dict[str, Any]:
    sels = [r["judge_selections"][judge_name] for r in rows]
    pairs = [(r.get("race"), r.get("recid_label"), s["pred"])
             for r, s in zip(rows, sels)]
    total_cands = sum(len(r["candidates"]) for r in rows)
    return {
        "n": len(rows),
        "accuracy": accuracy_of([s["is_correct"] for s in sels]),
        **fairness_block(pairs),
        "candidate_pass_rate": pct(sum(s["num_passed"] for s in sels), total_cands),
        "questions_with_a_passing_candidate": pct(
            sum(1 for s in sels if s["num_passed"] > 0), len(sels)),
        "fallback_used_pct": pct(sum(1 for s in sels if s["fallback_used"]), len(sels)),
        "changed_vs_majority_all_pct": pct(
            sum(1 for s in sels if s["changed_vs_majority_all"]), len(sels)),
        "unparseable_candidate_pct": pct(
            sum(s["num_unparseable"] for s in sels), total_cands),
    }


def summarize_baseline(rows: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    ans_f = BASELINES[name]
    correct_flags, pairs = [], []
    for r in rows:
        ans = r.get(ans_f, "") or ""
        correct = r.get("correct_answer")
        correct_flags.append((ans == correct) if correct else None)
        pairs.append((r.get("race"), r.get("recid_label"),
                      1 if ans == HIGH else 0))
    return {"n": len(rows), "accuracy": accuracy_of(correct_flags),
            **fairness_block(pairs)}


def judge_agreement(rows: List[Dict[str, Any]], names: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for a, b in itertools.combinations(names, 2):
        both = agree = 0
        for r in rows:
            for c in r["candidates"]:
                ea = c.get("judge_scores", {}).get(a, {})
                eb = c.get("judge_scores", {}).get(b, {})
                if ea.get("score") is None or eb.get("score") is None:
                    continue
                both += 1
                agree += ea.get("passed") == eb.get("passed")
        q_same = sum(
            1 for r in rows
            if r["judge_selections"][a]["answer"] == r["judge_selections"][b]["answer"]
        )
        out[f"{a} vs {b}"] = {
            "candidate_pass_agreement_pct": pct(agree, both),
            "candidates_compared": both,
            "final_answer_agreement_pct": pct(q_same, len(rows)),
        }
    return out


# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    overrides: Dict[str, float] = {}
    for spec in args.pass_score:
        if "=" not in spec:
            raise SystemExit(f"--pass_score expects name=value, got '{spec}'")
        name, val = spec.split("=", 1)
        overrides[name.strip()] = float(val)

    path = find_result_file(args.input)
    with open(path) as f:
        payload = json.load(f)
    rows = payload["results"] if isinstance(payload, dict) else payload
    meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    best_of_n = meta.get("best_of_n", len(rows[0].get("candidates", [])) or 1)
    registry = meta.get("judges", {})

    missing = [i for i, r in enumerate(rows) if not r.get("candidates")]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} rows have no 'candidates'.")

    judge_names = sorted({n for r in rows for c in r["candidates"]
                          for n in c.get("judge_scores", {})})
    if not judge_names:
        raise SystemExit(
            f"{path}: no judge_scores found on any candidate — run "
            f"judge_compas_candidates.py first."
        )
    if args.judges:
        unknown = [j for j in args.judges if j not in judge_names]
        if unknown:
            raise SystemExit(f"Judges not found in file: {unknown} "
                             f"(available: {judge_names})")
        judge_names = args.judges
    bad_overrides = [j for j in overrides if j not in judge_names]
    if bad_overrides:
        raise SystemExit(f"--pass_score for unknown judges: {bad_overrides} "
                         f"(available: {judge_names})")

    have_labels = any(r.get("recid_label") is not None for r in rows)
    if not have_labels:
        print("WARNING: no recid labels in results — accuracy / EOpp / EO will "
              "be n/a; only the DP gap is label-free.")

    print(f"Result file: {path}")
    print(f"Judges: {', '.join(judge_names)}"
          + (f" | threshold overrides: {overrides}" if overrides else ""))

    # ---- selection per judge + per-method row enrichment -------------------- #
    for r in rows:
        r["judge_selections"] = {
            name: select_for_judge(r, name, args.no_pass_fallback,
                                   overrides.get(name))
            for name in judge_names
        }
        label = r.get("recid_label")
        for method, field in BASELINES.items():
            ans = r.get(field, "") or ""
            r[f"{method}_pred"] = 1 if ans == HIGH else 0
        for name in judge_names:
            r[f"filtered_{name}_pred"] = r["judge_selections"][name]["pred"]

        def err(pred):
            if label is None:
                return None
            if int(label) == 1:
                return "TP" if pred == 1 else "FN"
            return "FP" if pred == 1 else "TN"

        for method in BASELINES:
            r[f"{method}_error_type"] = err(r[f"{method}_pred"])
        for name in judge_names:
            r[f"filtered_{name}_error_type"] = err(r[f"filtered_{name}_pred"])

    # ---- summaries ---------------------------------------------------------- #
    n = len(rows)
    baselines = {name: summarize_baseline(rows, name) for name in BASELINES}
    baselines["oracle_pass_at_n"] = {
        "accuracy": pct(sum(1 for r in rows if r.get("oracle_is_correct")), n)
        if have_labels else None
    }
    per_judge = {name: summarize_judge(rows, name, best_of_n) for name in judge_names}
    agreement = judge_agreement(rows, judge_names)

    stats = {
        "input": path,
        "num_defendants": n,
        "best_of_n": best_of_n,
        "no_pass_fallback": args.no_pass_fallback,
        "labels_available": have_labels,
        "judges": {name: {**registry.get(name, {}),
                          **({"pass_score_override": overrides[name]}
                             if name in overrides else {})}
                   for name in judge_names},
        "baselines": baselines,
        "per_judge": per_judge,
        "judge_agreement": agreement,
    }

    out_dir = args.output_dir or (
        args.input if os.path.isdir(args.input) else os.path.dirname(path) or "."
    )
    os.makedirs(out_dir, exist_ok=True)

    compared_path = os.path.join(out_dir, "compas_results_compared.json")
    with open(compared_path, "w") as f:
        json.dump(payload, f, indent=2)
    stats_path = os.path.join(out_dir, "judge_comparison_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ---- CSV ---------------------------------------------------------------- #
    def rnd(v):
        return round(v, 2) if v is not None else None

    csv_rows = []

    def csv_row(method, summ, extra=None):
        row = {"method": method, "n": summ["n"],
               "accuracy": rnd(summ["accuracy"]),
               "eopp_tpr_gap": rnd(summ["eopp_tpr_gap"]),
               "fpr_gap": rnd(summ["fpr_gap"]),
               "eo_gap": rnd(summ["eo_gap"]),
               "dp_gap": rnd(summ["dp_gap"])}
        row.update(extra or {})
        csv_rows.append(row)

    for bname in BASELINES:
        csv_row(bname, baselines[bname])
    for jname in judge_names:
        s = per_judge[jname]
        csv_row(f"filtered:{jname}", s, {
            "candidate_pass_rate": rnd(s["candidate_pass_rate"]),
            "fallback_used_pct": rnd(s["fallback_used_pct"]),
        })

    csv_path = os.path.join(out_dir, "judge_comparison.csv")
    fieldnames = ["method", "n", "accuracy", "eopp_tpr_gap", "fpr_gap",
                  "eo_gap", "dp_gap", "candidate_pass_rate", "fallback_used_pct"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # ---- report ------------------------------------------------------------- #
    def f_(v, w=9):
        return f"{v:>{w - 1}.2f}%" if v is not None else f"{'n/a':>{w}}"

    print(f"\n{'=' * 92}")
    print(f"METHOD COMPARISON  ({n} defendants x {best_of_n} candidates)")
    print(f"{'=' * 92}")
    hdr = (f"\n  {'Method':<26}{'Acc':>9}{'EOpp':>9}{'FPRgap':>9}{'EO':>9}"
           f"{'DPgap':>9}{'PassRate':>10}{'Fallback':>10}")
    print(hdr)
    print(f"  {'-' * (len(hdr) + 2)}")

    def line(name, s, extra=""):
        print(f"  {name:<26}{f_(s['accuracy'])}{f_(s['eopp_tpr_gap'])}"
              f"{f_(s['fpr_gap'])}{f_(s['eo_gap'])}{f_(s['dp_gap'])}{extra}")

    for bname in BASELINES:
        line(bname, baselines[bname])
    for jname in judge_names:
        s = per_judge[jname]
        line(f"filtered:{jname}", s,
             f"{f_(s['candidate_pass_rate'], 10)}{f_(s['fallback_used_pct'], 10)}")
    if baselines["oracle_pass_at_n"]["accuracy"] is not None:
        print(f"  {'oracle (pass@N)':<26}"
              f"{f_(baselines['oracle_pass_at_n']['accuracy'])}")

    races = sorted({r.get("race") for r in rows if r.get("race")})
    print(f"\n  Per-race TPR | FPR per method:")
    print(f"    {'Method':<24}" + "".join(f"{race[:20]:>24}" for race in races))
    for name, s in [*baselines.items(), *(
            (f"filtered:{j}", per_judge[j]) for j in judge_names)]:
        if "by_race" not in s:
            continue
        cells = ""
        for race in races:
            v = s["by_race"].get(race, {})
            cells += f"{f_(v.get('tpr'), 12)}|{f_(v.get('fpr'), 11)}"
        print(f"    {name:<24}{cells}")

    if len(judge_names) > 1:
        print(f"\n  Pairwise agreement (candidate pass / final answer):")
        for pair, a in agreement.items():
            print(f"    {pair:<38}{f_(a['candidate_pass_agreement_pct'], 8)} / "
                  f"{f_(a['final_answer_agreement_pct'], 8)}")

    print(f"\nCompared file: {compared_path}")
    print(f"Stats saved to: {stats_path}")
    print(f"CSV saved to:   {csv_path}")
    print("Complete!")


if __name__ == "__main__":
    main()