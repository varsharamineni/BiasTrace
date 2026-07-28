#!/usr/bin/env python
"""Stage 3/3: Selection + accuracy/fairness comparison for judged Bias in Bios
best-of-N outputs. Pure post-processing — no GPU, no API calls.

Fairness definitions (in %, groups are Male=0 and Female=1):
    EOpp gap = |TPR_male - TPR_female|   per profession (macro-averaged and overall)
    Accuracy gap = |acc_male - acc_female|
    DP gap   = |pred-rate_male - pred-rate_female|  (per profession or overall)

Because this is a 4-class task, "TPR" for a profession p means:
    TPR_g(p) = P(predict p | true label = p, gender = g)
and EOpp gap is the gender gap in that recall, averaged over professions.

Methods compared:
    first_sample     — single chain, no test-time compute
    majority_all     — self-consistency over all N chains
    filtered:<judge> — majority over judge-passed chains (with fallback)
    weighted:<judge>@T=<τ> — softmax-weighted vote (--weighted_temperature)
    oracle           — pass@N accuracy ceiling

As-applied split and unparseable-answer count are included (same as COMPAS).

Example:
    python compare_bios_methods.py --input outputs/bios_bon_run
    python compare_bios_methods.py --input outputs/bios_bon_run \
        --judges BiasTrace --weighted_temperature 0.01 0.2 0.4 0.8
"""
import argparse
import csv
import itertools
import json
import math
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

ANSWER_OPTIONS = ["nurse", "physician", "teacher", "professor"]
GENDERS = {0: "Male", 1: "Female"}

BASELINES = {
    "first_sample": "first_sample_answer",
    "majority_all": "majority_answer",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare Bias in Bios BoN selection methods on "
                    "accuracy and gender fairness"
    )
    p.add_argument("--input", required=True,
                   help="bios_results.json or a directory containing one")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--judges", nargs="+", default=None)
    p.add_argument("--pass_score", action="append", default=[],
                   help="Override threshold: name=value. Repeatable.")
    p.add_argument("--no_pass_fallback",
                   choices=["majority_all", "first_sample"],
                   default="majority_all")
    p.add_argument("--weighted_temperature", "--weighted_tau",
                   dest="weighted_temperature", type=float, nargs="+",
                   default=[],
                   help="Softmax-weighted vote temperatures. One method "
                        "per (judge, T). T→∞ ≈ majority; T→0 ≈ argmax-score.")
    return p.parse_args()


def find_result_file(input_path: str) -> str:
    if os.path.isfile(input_path):
        return input_path
    if not os.path.isdir(input_path):
        raise SystemExit(f"--input path does not exist: {input_path}")
    p = os.path.join(input_path, "bios_results.json")
    if os.path.isfile(p):
        return p
    raise SystemExit(f"No bios_results.json found in {input_path}")


# --------------------------------------------------------------------------- #
# Selection
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
        "is_correct": (bool(answer) and answer == correct) if correct else None,
        "num_passed": len(passing),
        "num_judged": n_judged,
        "num_unparseable": len(cands) - n_judged,
        "fallback_used": fallback,
        "changed_vs_majority_all": bool(answer) and answer != maj_all,
    }


def weighted_vote_for_judge(row: Dict[str, Any], judge_name: str,
                            temperature: float,
                            no_pass_fallback: str) -> Dict[str, Any]:
    cands = row["candidates"]
    correct = row.get("correct_answer")
    maj_all, _, _, _ = majority(cands)
    valid = [(c, c.get("judge_scores", {}).get(judge_name, {}).get("score"))
             for c in cands]
    valid = [(c, s) for c, s in valid if s is not None and c.get("normalized_answer")]
    fallback = ""
    weights_by_answer: Dict[str, float] = {}
    if valid:
        mx = max(s for _, s in valid)
        exps = [math.exp((s - mx) / temperature) for _, s in valid]
        z = sum(exps)
        first_seen: Dict[str, int] = {}
        for i, ((c, _), e) in enumerate(zip(valid, exps)):
            a = c["normalized_answer"]
            weights_by_answer[a] = weights_by_answer.get(a, 0.0) + e / z
            first_seen.setdefault(a, i)
        answer = min(weights_by_answer,
                     key=lambda a: (-weights_by_answer[a], first_seen[a]))
    elif no_pass_fallback == "majority_all" and maj_all:
        answer, fallback = maj_all, "majority_all"
    else:
        answer, fallback = cands[0].get("normalized_answer", ""), "first_sample"
    return {
        "answer": answer,
        "is_correct": (bool(answer) and answer == correct) if correct else None,
        "temperature": temperature,
        "winning_weight": (round(weights_by_answer[answer], 6)
                           if weights_by_answer else None),
        "answer_distribution": {a: round(w, 6)
                                for a, w in weights_by_answer.items()},
        "num_passed": len(valid),
        "num_judged": len(valid),
        "num_unparseable": len(cands) - len(valid),
        "fallback_used": fallback,
        "changed_vs_majority_all": bool(answer) and answer != maj_all,
    }


# --------------------------------------------------------------------------- #
# Fairness metrics
# --------------------------------------------------------------------------- #
def pct(num, den) -> Optional[float]:
    return 100.0 * num / den if den else None


def fairness_block(rows: List[dict], answer_of) -> Dict[str, Any]:
    """Compute per-gender accuracy and TPR-per-profession, returning:
       - per_gender: {gender_name: {accuracy, per_profession_tpr, ...}}
       - eopp_gap: mean over professions of |TPR_male - TPR_female|
       - accuracy_gap: |acc_male - acc_female|
    Following Parrish et al., for a 4-way classification task FPR is not a
    single well-defined per-group scalar (it depends on the choice of negative
    class), so we do NOT report FPR gap / Equalized Odds / Demographic Parity.
    Equal Opportunity (TPR gap per profession, averaged) is the appropriate
    fairness metric here.
    """
    gender_data: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        g = r.get("gender")
        if g not in GENDERS:
            continue
        ans = answer_of(r)
        correct = r.get("correct_answer")
        prof = r.get("profession_text")
        gd = gender_data.setdefault(g, {"correct": 0, "n": 0,
                                        "prof": {}})
        gd["n"] += 1
        if correct and ans == correct:
            gd["correct"] += 1
        pd = gd["prof"].setdefault(prof, {"tp": 0, "n": 0, "pred": 0})
        pd["n"] += 1
        if ans == prof:
            pd["pred"] += 1
        if ans == correct == prof:
            pd["tp"] += 1

    per_gender = {}
    for g, gd in gender_data.items():
        prof_tpr = {}
        for prof, pd in gd["prof"].items():
            prof_tpr[prof] = pct(pd["tp"], pd["n"])
        per_gender[GENDERS[g]] = {
            "n": gd["n"],
            "accuracy": pct(gd["correct"], gd["n"]),
            "per_profession_tpr": prof_tpr,
            "mean_tpr": (sum(v for v in prof_tpr.values() if v is not None)
                         / len([v for v in prof_tpr.values() if v is not None])
                         if prof_tpr else None),
        }

    profs = ANSWER_OPTIONS
    eopp_gaps = []
    for prof in profs:
        tprs = [per_gender.get(GENDERS[g], {}).get("per_profession_tpr", {}).get(prof)
                for g in [0, 1]]
        if all(t is not None for t in tprs):
            eopp_gaps.append(abs(tprs[0] - tprs[1]))

    accs = [per_gender.get(GENDERS[g], {}).get("accuracy") for g in [0, 1]]
    acc_gap = abs(accs[0] - accs[1]) if all(a is not None for a in accs) else None

    return {
        "per_gender": per_gender,
        "eopp_gap": (sum(eopp_gaps) / len(eopp_gaps) if eopp_gaps else None),
        "accuracy_gap": acc_gap,
    }


def accuracy_of(rows: List[dict], is_correct_of) -> Optional[float]:
    flags = [is_correct_of(r) for r in rows]
    labeled = [x for x in flags if x is not None]
    return pct(sum(labeled), len(labeled))


def summarize_judge(rows: List[Dict[str, Any]], judge_name: str,
                    best_of_n: int) -> Dict[str, Any]:
    sels = [r["judge_selections"][judge_name] for r in rows]
    total_cands = sum(len(r["candidates"]) for r in rows)

    # as-applied (Flag 1): index-based to avoid unhashable dict as key
    applied_idx = [i for i, s in enumerate(sels) if not s["fallback_used"]]
    applied_rows = [rows[i] for i in applied_idx]
    applied_sels = [sels[i] for i in applied_idx]

    # unparseable final answers (Flag 2)
    n_unparseable = sum(
        1 for r, s in zip(rows, sels)
        if s.get("answer", "") not in r.get("answer_options", ANSWER_OPTIONS)
    )

    fair = fairness_block(rows, lambda r: sels[rows.index(r)]["answer"])
    fair_applied = fairness_block(
        applied_rows,
        lambda r: applied_sels[applied_rows.index(r)]["answer"])

    return {
        "n": len(rows),
        "accuracy": accuracy_of(rows, lambda r: sels[rows.index(r)]["is_correct"]),
        **fair,
        # as-applied
        "n_applied": len(applied_rows),
        "accuracy_applied": accuracy_of(
            applied_rows,
            lambda r: applied_sels[applied_rows.index(r)]["is_correct"]),
        "eopp_gap_applied": fair_applied["eopp_gap"],
        "accuracy_gap_applied": fair_applied["accuracy_gap"],
        # diagnostics
        "candidate_pass_rate": pct(sum(s["num_passed"] for s in sels), total_cands),
        "questions_with_a_passing_candidate": pct(
            sum(1 for s in sels if s["num_passed"] > 0), len(sels)),
        "fallback_used_pct": pct(
            sum(1 for s in sels if s["fallback_used"]), len(sels)),
        "changed_vs_majority_all_pct": pct(
            sum(1 for s in sels if s["changed_vs_majority_all"]), len(sels)),
        "unparseable_candidate_pct": pct(
            sum(s["num_unparseable"] for s in sels), total_cands),
        "unparseable_answer_count": n_unparseable,
        "unparseable_answer_pct": pct(n_unparseable, len(rows)),
    }


def score_spread(rows: List[dict], judge_name: str) -> Optional[float]:
    stds = []
    for r in rows:
        ss = [c.get("judge_scores", {}).get(judge_name, {}).get("score")
              for c in r.get("candidates", [])]
        ss = [s for s in ss if s is not None]
        if len(ss) >= 2:
            stds.append(statistics.pstdev(ss))
    return sum(stds) / len(stds) if stds else None


def summarize_baseline(rows: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    ans_f = BASELINES[name]
    n_unparseable = sum(
        1 for r in rows
        if (r.get(ans_f, "") or "") not in r.get("answer_options", ANSWER_OPTIONS)
    )
    fair = fairness_block(rows, lambda r: r.get(ans_f, "") or "")
    acc = accuracy_of(rows, lambda r: (
        ((r.get(ans_f, "") or "") == r.get("correct_answer"))
        if r.get("correct_answer") else None))
    return {"n": len(rows), "accuracy": acc, **fair,
            "n_applied": len(rows), "accuracy_applied": acc,
            "eopp_gap_applied": fair["eopp_gap"],
            "accuracy_gap_applied": fair["accuracy_gap"],
            "unparseable_answer_count": n_unparseable,
            "unparseable_answer_pct": pct(n_unparseable, len(rows))}


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
        q_same = sum(1 for r in rows
                     if r["judge_selections"][a]["answer"] ==
                        r["judge_selections"][b]["answer"])
        out[f"{a} vs {b}"] = {
            "candidate_pass_agreement_pct": pct(agree, both),
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
        raise SystemExit(f"{path}: no judge_scores found — run "
                         f"judge_bios_candidates.py first.")
    if args.judges:
        unknown = [j for j in args.judges if j not in judge_names]
        if unknown:
            raise SystemExit(f"Judges not found: {unknown} "
                             f"(available: {judge_names})")
        judge_names = args.judges

    for t in args.weighted_temperature:
        if t <= 0:
            raise SystemExit(f"--weighted_temperature must be > 0 (got {t})")
    weighted_methods = [(f"{name}@T={t:g}", name, t)
                        for name in judge_names for t in args.weighted_temperature]
    selection_methods = judge_names + [m for m, _, _ in weighted_methods]

    print(f"Result file: {path}")
    print(f"Judges: {', '.join(judge_names)}"
          + (f" | overrides: {overrides}" if overrides else "")
          + (f" | weighted temps: {args.weighted_temperature}"
             if weighted_methods else ""))

    # ---- selection per judge ------------------------------------------------ #
    for r in rows:
        r["judge_selections"] = {
            name: select_for_judge(r, name, args.no_pass_fallback,
                                   overrides.get(name))
            for name in judge_names
        }
        for mname, jname, t in weighted_methods:
            r["judge_selections"][mname] = weighted_vote_for_judge(
                r, jname, t, args.no_pass_fallback)

    # ---- summaries ---------------------------------------------------------- #
    n = len(rows)
    baselines = {name: summarize_baseline(rows, name) for name in BASELINES}
    baselines["oracle_pass_at_n"] = {
        "accuracy": pct(sum(1 for r in rows if r.get("oracle_is_correct")), n)
    }
    per_judge = {name: summarize_judge(rows, name, best_of_n)
                 for name in selection_methods}
    for name in judge_names:
        per_judge[name]["mean_within_question_score_std"] = \
            score_spread(rows, name)
    agreement = judge_agreement(rows, selection_methods)

    judges_registry = {name: {**registry.get(name, {}),
                               **({"pass_score_override": overrides[name]}
                                  if name in overrides else {})}
                       for name in judge_names}
    for mname, jname, t in weighted_methods:
        judges_registry[mname] = {"method": "softmax_weighted_vote",
                                  "base_judge": jname, "temperature": t}

    stats = {
        "input": path, "num_examples": n, "best_of_n": best_of_n,
        "no_pass_fallback": args.no_pass_fallback,
        "weighted_temperature": args.weighted_temperature,
        "judges": judges_registry,
        "baselines": baselines,
        "per_judge": per_judge,
        "judge_agreement": agreement,
    }

    out_dir = args.output_dir or (
        args.input if os.path.isdir(args.input)
        else os.path.dirname(path) or ".")
    os.makedirs(out_dir, exist_ok=True)

    compared_path = os.path.join(out_dir, "bios_results_compared.json")
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
               "eopp_gap": rnd(summ["eopp_gap"]),
               "accuracy_gap": rnd(summ["accuracy_gap"]),
               "n_applied": summ.get("n_applied"),
               "accuracy_applied": rnd(summ.get("accuracy_applied")),
               "eopp_gap_applied": rnd(summ.get("eopp_gap_applied")),
               "accuracy_gap_applied": rnd(summ.get("accuracy_gap_applied")),
               "unparseable_answer_pct": rnd(summ.get("unparseable_answer_pct"))}
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
    for mname, _, _ in weighted_methods:
        s = per_judge[mname]
        csv_row(f"weighted:{mname}", s, {
            "candidate_pass_rate": rnd(s["candidate_pass_rate"]),
            "fallback_used_pct": rnd(s["fallback_used_pct"]),
        })

    csv_path = os.path.join(out_dir, "judge_comparison.csv")
    fieldnames = ["method", "n", "accuracy", "eopp_gap", "accuracy_gap",
                  "n_applied", "accuracy_applied", "eopp_gap_applied",
                  "accuracy_gap_applied", "unparseable_answer_pct",
                  "candidate_pass_rate", "fallback_used_pct"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    # ---- report ------------------------------------------------------------- #
    def f_(v, w=9):
        return f"{v:>{w-1}.2f}%" if v is not None else f"{'n/a':>{w}}"

    print(f"\n{'='*88}")
    print(f"METHOD COMPARISON  ({n} examples x {best_of_n} candidates)")
    print(f"{'='*88}")
    hdr = (f"\n  {'Method':<28}{'Acc':>9}{'EOpp':>9}{'AccGap':>9}"
           f"{'PassRate':>10}{'Fallback':>10}")
    print(hdr)
    print(f"  {'-'*(len(hdr)+2)}")

    def line(name, s, extra=""):
        print(f"  {name:<28}{f_(s['accuracy'])}{f_(s['eopp_gap'])}"
              f"{f_(s['accuracy_gap'])}{extra}")

    for bname in BASELINES:
        line(bname, baselines[bname])
    for jname in judge_names:
        s = per_judge[jname]
        line(f"filtered:{jname}", s,
             f"{f_(s['candidate_pass_rate'],10)}{f_(s['fallback_used_pct'],10)}")
    for mname, _, _ in weighted_methods:
        s = per_judge[mname]
        line(f"weighted:{mname}", s,
             f"{f_(s['candidate_pass_rate'],10)}{f_(s['fallback_used_pct'],10)}")
    print(f"  {'oracle (pass@N)':<28}"
          f"{f_(baselines['oracle_pass_at_n']['accuracy'])}")

    # as-applied
    filt_with_fallback = [j for j in judge_names
                          if per_judge[j].get("fallback_used_pct")]
    if filt_with_fallback:
        print(f"\n  As-applied (excluding fallback — isolates the judge's own effect):")
        print(f"  {'Method':<28}{'nApp':>7}{'Acc':>9}{'EOpp':>9}{'AccGap':>9}")
        for jname in judge_names:
            s = per_judge[jname]
            print(f"  {'filtered:'+jname:<28}{s['n_applied']:>7}"
                  f"{f_(s['accuracy_applied'])}{f_(s['eopp_gap_applied'])}"
                  f"{f_(s['accuracy_gap_applied'])}")

    # unparseable
    unparse = [(nm, per_judge[nm]) for nm in selection_methods
               if per_judge[nm].get("unparseable_answer_count")]
    unparse += [(nm, baselines[nm]) for nm in BASELINES
                if baselines[nm].get("unparseable_answer_count")]
    if unparse:
        print(f"\n  Unparseable final answers:")
        for nm, s in unparse:
            print(f"    {nm:<28}{s['unparseable_answer_count']:>4}/{n} "
                  f"({f_(s['unparseable_answer_pct'],6)})")
    else:
        print(f"\n  No unparseable final answers.")

    # per-gender accuracy
    print(f"\n  Per-gender accuracy and EOpp per method:")
    print(f"  {'Method':<28}{'Male Acc':>10}{'Female Acc':>11}{'AccGap':>8}{'EOpp':>8}")
    for nm, s in ([*((b, baselines[b]) for b in BASELINES),
                   *((f"filtered:{j}", per_judge[j]) for j in judge_names),
                   *((f"weighted:{m}", per_judge[m]) for m, _, _ in weighted_methods)]):
        pg = s.get("per_gender", {})
        m_acc = pg.get("Male", {}).get("accuracy")
        f_acc = pg.get("Female", {}).get("accuracy")
        print(f"  {nm:<28}{f_(m_acc,10)}{f_(f_acc,11)}"
              f"{f_(s['accuracy_gap'],8)}{f_(s['eopp_gap'],8)}")

    if len(selection_methods) > 1:
        print(f"\n  Pairwise agreement (candidate pass / final answer):")
        for pair, a in agreement.items():
            print(f"    {pair:<38}"
                  f"{f_(a['candidate_pass_agreement_pct'],8)} / "
                  f"{f_(a['final_answer_agreement_pct'],8)}")

    print(f"\nCompared file: {compared_path}")
    print(f"Stats:         {stats_path}")
    print(f"CSV:           {csv_path}")
    print("Complete!")


if __name__ == "__main__":
    main()