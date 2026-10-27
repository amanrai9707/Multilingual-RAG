"""Turn the scored sheets into the results tables and hypothesis tests (Protocol sections 10 and 13).

Step 1 - after both raters have scored their own copy of results/annotation.csv:
    python analyze_results.py --make-resolved-template --rater-a scoredA.csv --rater-b scoredB.csv --out results/analysis
  -> writes resolved_template.csv: agreed scores filled in, DISAGREEMENTS left blank. Settle each blank by going back to the
     source passage (never average), fill it in, save as resolved.csv.
Step 2:
    python analyze_results.py --log results/raw_log.jsonl --key results/key.csv --dataset <workbook/csv> \
        --rater-a scoredA.csv --rater-b scoredB.csv --resolved resolved.csv --out results/analysis
  -> results_report.md, results_summary.json
"""
from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis.stats import (weighted_kappa, paired_wilcoxon, bootstrap_ci_mean_diff, sensitivity_wilcoxon, recall_at_k,
                            mean_reciprocal_rank, hallucination_rate, unsupported_claim_rate, false_abstention_rate,
                            h3_degradation_test)
from src.dataset_io import load_dataset, gold_keys

COLS = ["rater_correctness_0_2", "rater_groundedness_0_2", "rater_has_unsupported_claim_0_1"]
SHORT = {"rater_correctness_0_2": "correctness", "rater_groundedness_0_2": "groundedness", "rater_has_unsupported_claim_0_1": "claim"}


def read_scores(path):
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[r["blind_id"]] = {SHORT[c]: (None if str(r.get(c, "")).strip() == "" else int(float(r[c]))) for c in COLS}
    return out


def kappas(a, b):
    from sklearn.metrics import cohen_kappa_score
    res = {}
    for m in ("correctness", "groundedness", "claim"):
        ids = [i for i in a if i in b and a[i][m] is not None and b[i][m] is not None]
        x, y = [a[i][m] for i in ids], [b[i][m] for i in ids]
        agree = float(np.mean([p == q for p, q in zip(x, y)])) if ids else None
        try:
            k = float(weighted_kappa(x, y)) if m != "claim" else float(cohen_kappa_score(x, y))
            k = None if np.isnan(k) else k
        except Exception:
            k = None
        res[m] = {"n": len(ids), "exact_agreement": agree, "kappa": k}
    return res


def make_template(a, b, out_csv):
    n_dis = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["blind_id", "correctness", "groundedness", "unsupported_claim", "disagreed_fields"])
        for i in sorted(a):
            row, dis = [i], []
            for m in ("correctness", "groundedness", "claim"):
                va, vb = a[i][m], b.get(i, {}).get(m)
                if va is not None and va == vb:
                    row.append(va)
                else:
                    row.append("")
                    dis.append(m)
            n_dis += bool(dis)
            w.writerow(row + [";".join(dis)])
    return n_dis


def fmt(x, nd=3):
    return "n/a" if x is None else ("%.*f" % (nd, x) if isinstance(x, float) else str(x))


def test_row(label, r, ci):
    return "| %s | %d | %s | %s | %s | %s [%s, %s] | %s |" % (
        label, r["n_used"], fmt(r["p_value"], 4), fmt(r["rank_biserial"], 2), fmt(r["mean_diff"], 3), fmt(ci[0], 3),
        fmt(ci[1], 3), fmt(ci[2], 3), r["n_tied"])


def analyze(args):
    rows = {r["question_id"]: r for r in load_dataset(args.dataset)}
    key = {r["blind_id"]: (r["question_id"], r["system"]) for r in csv.DictReader(open(args.key, newline="", encoding="utf-8"))}
    log = {}
    for line in open(args.log, encoding="utf-8"):
        rec = json.loads(line)
        log[(rec["question_id"], rec["system"])] = rec
    res_scores = {}
    for r in csv.DictReader(open(args.resolved, newline="", encoding="utf-8-sig")):
        vals = [r["correctness"], r["groundedness"], r["unsupported_claim"]]
        if any(str(v).strip() == "" for v in vals):
            raise SystemExit("resolved.csv still has blanks (blind_id %s). Settle every disagreement first." % r["blind_id"])
        res_scores[r["blind_id"]] = tuple(int(float(v)) for v in vals)
    k = next(iter(log.values()))["config"]["k"]

    items = []
    for bid, (qid, sysname) in key.items():
        c, g, u = res_scores[bid]
        rec, row = log[(qid, sysname)], rows[qid]
        gold = gold_keys(row)
        got = ["%s|%s" % (ch["doc_id"], ch["page"]) for ch in rec["retrieved"]]
        items.append({"question_id": qid, "system": sysname, "correctness": c, "groundedness": g, "claim": u,
                      "abstained": bool(rec["abstained"]), "retrieved": got, "gold": gold,
                      "answerable": bool(set(gold) & set(got[:k])), "tags": row["terminology_tags"], "grade": row["evidence_grade"]})
    df = pd.DataFrame(items)

    summary = {"k": k, "systems": {}, "kappa": None, "tests": {}}
    for s, d in df.groupby("system"):
        e = {"n": int(len(d)), "mean_correctness": float(d.correctness.mean()), "share_correct_2": float((d.correctness == 2).mean()),
             "mean_groundedness": float(d.groundedness.mean()), "unsupported_answer_rate": hallucination_rate(d.groundedness),
             "unsupported_claim_rate": unsupported_claim_rate(d.claim), "abstain_rate": float(d.abstained.mean())}
        if s in ("B", "C"):
            g = d[d.gold.apply(len) > 0]          # news-sourced questions have no page, so they cannot be scored for retrieval
            e["n_with_gold_page"] = int(len(g))
            e["recall_at_k"] = recall_at_k(list(g.retrieved), list(g.gold), k) if len(g) else None
            e["mrr"] = mean_reciprocal_rank(list(g.retrieved), list(g.gold)) if len(g) else None
            e["false_abstention_rate"] = false_abstention_rate(list(g.abstained), list(g.answerable)) if len(g) else None
        summary["systems"][s] = e

    wide = df.pivot(index="question_id", columns="system", values=["correctness", "groundedness", "claim"])
    wide.columns = ["%s_%s" % (m, s) for m, s in wide.columns]
    meta = pd.DataFrame({q: {"tags": r["terminology_tags"], "evidence_grade": r["evidence_grade"]} for q, r in rows.items()}).T
    wide = wide.join(meta, how="left")

    def comp(name, m, x, y):
        cx, cy = "%s_%s" % (m, x), "%s_%s" % (m, y)
        d = wide.dropna(subset=[cx, cy])
        sens = sensitivity_wilcoxon(d, cx, cy)
        return {"measure": m, "n_pairs": int(len(d)), "all": sens["all"], "primary_only": sens["primary_only"], "n_primary": sens["n_primary"],
                "ci_all": bootstrap_ci_mean_diff(d[cx], d[cy])}

    summary["tests"]["H1_B_vs_C"] = [comp("H1", m, "B", "C") for m in ("correctness", "groundedness")]
    summary["tests"]["H2_A_vs_B"] = [comp("H2", m, "A", "B") for m in ("groundedness", "claim", "correctness")]
    h3 = {}
    for m in ("correctness", "groundedness"):
        d = wide.dropna(subset=["%s_B" % m, "%s_C" % m])
        h3[m] = h3_degradation_test(d, "%s_B" % m, "%s_C" % m, "tags")
    summary["tests"]["H3_terminology_vs_general"] = h3
    if args.rater_a and args.rater_b:
        summary["kappa"] = kappas(read_scores(args.rater_a), read_scores(args.rater_b))
    return summary


def report_md(s):
    L = ["# Results report (generated)", "", "k = %d. **Remember:** these numbers are only as good as the frozen dataset, the blinded scoring and "
         "the agreed protocol. Translation quality is evaluated separately (Protocol section 14)." % s["k"], "", "## Per-system results", "",
         "| System | n | Mean correctness | % score 2 | Mean groundedness | Unsupported-answer rate | Unsupported-claim rate | Abstain rate | Recall@k | MRR | False-abstention |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for sysname, e in sorted(s["systems"].items()):
        L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            sysname, e["n"], fmt(e["mean_correctness"], 2), fmt(e["share_correct_2"], 2), fmt(e["mean_groundedness"], 2),
            fmt(e["unsupported_answer_rate"], 2), fmt(e["unsupported_claim_rate"], 2), fmt(e["abstain_rate"], 2),
            fmt(e.get("recall_at_k"), 2), fmt(e.get("mrr"), 2), fmt(e.get("false_abstention_rate"), 2)))
    if s["kappa"]:
        L += ["", "## Rater agreement", "", "| Measure | n | Exact agreement | Kappa |", "|---|---|---|---|"]
        for m, v in s["kappa"].items():
            L.append("| %s | %d | %s | %s |" % (m, v["n"], fmt(v["exact_agreement"], 2), fmt(v["kappa"], 2)))
    head = ["| Comparison | pairs that differ | p (Wilcoxon) | effect (rank-biserial) | mean diff [95% CI] | ties |", "|---|---|---|---|---|---|"]
    for title, key in (("H1: System B vs System C (does translation cost us?)", "H1_B_vs_C"), ("H2: System A vs System B (does retrieval help?)", "H2_A_vs_B")):
        L += ["", "## " + title, "", "Difference = first system minus second. Each measure is tested on ALL questions, then on PRIMARY-sourced questions only.", ""] + head
        for t in s["tests"][key]:
            L.append(test_row("%s (all, n=%d)" % (t["measure"], t["n_pairs"]), t["all"], t["ci_all"]))
            if t["primary_only"]:
                L.append("| %s (primary only, n=%d) | %d | %s | %s | %s | %s |" % (t["measure"], t["n_primary"], t["primary_only"]["n_used"],
                         fmt(t["primary_only"]["p_value"], 4), fmt(t["primary_only"]["rank_biserial"], 2), fmt(t["primary_only"]["mean_diff"], 3), t["primary_only"]["n_tied"]))
    L += ["", "## H3: do terminology questions degrade more under translation?", "",
          "Degradation = score(B) - score(C). Terminology group = any tag 1-5; general group = tag 6 only. Mann-Whitney U (two-sided, as proposed - confirm it was frozen before the run).", "",
          "| Measure | n terminology | n general | mean degradation (terminology) | mean degradation (general) | p | effect (rank-biserial) |", "|---|---|---|---|---|---|---|"]
    for m, v in s["tests"]["H3_terminology_vs_general"].items():
        L.append("| %s | %d | %d | %s | %s | %s | %s |" % (m, v["n_terminology"], v["n_general"], fmt(v.get("mean_degradation_terminology"), 3),
                 fmt(v.get("mean_degradation_general"), 3), fmt(v.get("p_value"), 4), fmt(v.get("rank_biserial"), 2)))
    L += ["", "## How to read this", "", "- A small p means the gap is unlikely to be chance; the **effect size and confidence interval** say whether it matters.",
          "- Report null results exactly as they come out (Protocol section 16).", "- If the all-questions and primary-only rows disagree, news-sourced questions are driving the result: say so."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log"), ap.add_argument("--key"), ap.add_argument("--dataset")
    ap.add_argument("--rater-a"), ap.add_argument("--rater-b"), ap.add_argument("--resolved")
    ap.add_argument("--out", default="results/analysis")
    ap.add_argument("--make-resolved-template", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.make_resolved_template:
        n = make_template(read_scores(a.rater_a), read_scores(a.rater_b), os.path.join(a.out, "resolved_template.csv"))
        print("Wrote %s. %d item(s) need a disagreement resolved by hand." % (os.path.join(a.out, "resolved_template.csv"), n))
        return 0
    for need in ("log", "key", "dataset", "resolved"):
        if not getattr(a, need):
            ap.error("--%s is required for the analysis" % need)
    s = analyze(a)
    json.dump(s, open(os.path.join(a.out, "results_summary.json"), "w"), indent=2, default=float)
    open(os.path.join(a.out, "results_report.md"), "w", encoding="utf-8").write(report_md(s))
    print("Wrote %s and results_summary.json" % os.path.join(a.out, "results_report.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
