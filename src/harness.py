"""Run the systems on the dataset, log everything, and export a BLINDED sheet for the raters.

BLINDING LIMIT (be honest about it in the paper): System C answers are in Assamese, so a rater can
tell C from A/B by language alone. True blinding therefore only holds within the A-vs-B comparison.
Mitigation: raters score each answer against the source passage, not against each other.
"""
from __future__ import annotations
import csv, json, random, time


def run_all(dataset, systems, config, log_path="results/raw_log.jsonl"):
    """dataset: list of dicts with question_id, english_question, assamese_question.
    systems: {"A": fn(row)->dict, "B": ..., "C": ...}. Appends one JSON line per (question, system)."""
    import os
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    records = []
    with open(log_path, "a", encoding="utf-8") as f:
        for row in dataset:
            for name, fn in systems.items():
                out = fn(row)
                rec = {"question_id": row["question_id"], "system": name, "config": config,
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **out}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                records.append(rec)
    return records


def export_blinded(records, dataset, annotation_csv, key_csv, seed=2026):
    """Shuffle rows, hide system names. annotation_csv goes to raters; key_csv stays with ONE person
    and is only opened after scoring is finished."""
    by_id = {r["question_id"]: r for r in dataset}
    rows = list(records)
    random.Random(seed).shuffle(rows)
    with open(annotation_csv, "w", newline="", encoding="utf-8") as fa, open(key_csv, "w", newline="", encoding="utf-8") as fk:
        wa, wk = csv.writer(fa), csv.writer(fk)
        wa.writerow(["blind_id", "question", "answer", "supporting_passage", "reference_answer",
                     "rater_correctness_0_2", "rater_groundedness_0_2", "rater_has_unsupported_claim_0_1"])
        wk.writerow(["blind_id", "question_id", "system"])
        for i, r in enumerate(rows, start=1):
            d = by_id[r["question_id"]]
            q = d["assamese_question"] if r["system"] == "C" else d["english_question"]
            wa.writerow([f"R{i:04d}", q, r["final_answer"], d.get("supporting_passage", ""), d.get("reference_answer", ""), "", "", ""])
            wk.writerow([f"R{i:04d}", r["question_id"], r["system"]])
