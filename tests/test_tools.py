"""End-to-end test of the practical tools, using NO models and NO internet.
Chain tested: real PDF -> pdf_to_clean -> validate dataset -> run_experiment --dry-run -> simulated raters ->
disagreement template -> analyze_results.        Run:  python tests/test_tools.py"""
import csv, json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from src.pdf_to_clean import convert
from src.dataset_io import load_dataset, validate, parse_pages, gold_keys, format_report
import run_experiment, analyze_results

N, PER_PAGE = 150, 6
rng = np.random.default_rng(11)
td = tempfile.mkdtemp()
facts = ["Under zq%d, the compensation rate is Rs %d per household." % (i, 1000 + 37 * i) for i in range(1, N + 1)]

# ---------------------------------------------------------------- 1. a real PDF with headers, footers and a scanned page
pdf = os.path.join(td, "doc.pdf")
c = canvas.Canvas(pdf, pagesize=A4)
n_pages = N // PER_PAGE
for pg in range(1, n_pages + 1):
    c.setFont("Helvetica", 9); c.drawString(60, 810, "Test Authority Report 2025 - Wildlife Conflict Rules")
    c.drawString(290, 30, str(pg))                                             # bare printed page number == PDF page
    c.setFont("Helvetica", 11)
    y = 740
    for f in facts[(pg - 1) * PER_PAGE: pg * PER_PAGE]:
        c.drawString(60, y, f); y -= 20
    c.showPage()
c.save()
clean = os.path.join(td, "clean")
rep = convert(pdf, "DOC-001", clean)
txt = open(os.path.join(clean, "DOC-001.txt"), encoding="utf-8").read()
assert rep["pages_written"] == n_pages and not rep["no_text_pages"], rep
assert "Test Authority Report" not in txt and "<<<PAGE 5>>>" in txt           # header removed, markers present
import re as _re
_norm = lambda t: _re.sub(r"[^a-z0-9]+", "", t.lower())
lost = [f for f in facts if _norm(f) not in _norm(txt)]
assert not lost, "converter silently deleted %d facts, e.g. %s" % (len(lost), lost[:2])      # NOTHING real may be removed
print("1. PDF -> clean text OK:", rep["pages_written"], "pages, running lines removed:", rep["running_lines_removed"])

# ---------------------------------------------------------------- 2. dataset + validator
fields = ["question_id", "source_document", "source_location", "underlying_fact", "terminology_tags", "english_question",
          "assamese_question", "reference_answer", "supporting_passage", "difficulty", "notes", "evidence_grade"]
patterns = ["1", "2", "3", "4", "5", "2,4", "1,5", "3,4"]
rows = []
for i in range(1, N + 1):
    tags = "6" if i <= 50 else patterns[(i - 51) % len(patterns)]
    primary = i <= 120
    rows.append({"question_id": "Q%03d" % i, "source_document": "DOC-001" if primary else "NEWS-001",
                 "source_location": "p.%d" % ((i - 1) // PER_PAGE + 1) if primary else "news article",
                 "underlying_fact": facts[i - 1], "terminology_tags": tags,
                 "english_question": "What is the compensation rate for zq%d?" % i,
                 "assamese_question": "\u0985\u09b8\u09ae\u09c0\u09af\u09bc\u09be \u09aa\u09cd\u09f0\u09b6\u09cd\u09a8 zq%d" % i,
                 "reference_answer": "Rs %d per household" % (1000 + 37 * i), "supporting_passage": facts[i - 1],
                 "difficulty": ["easy", "medium", "hard"][i % 3], "notes": "", "evidence_grade": "primary" if primary else "news-2plus"})
ds = os.path.join(td, "questions.csv")
with open(ds, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
res = validate(load_dataset(ds), clean)
assert not res["errors"], res["errors"][:5]
assert parse_pages("pp. 12-14") == ["12", "13", "14"] and parse_pages("12") == ["12"] and gold_keys(rows[0]) == ["DOC-001|1"]
print("2a. clean dataset validates: %d questions, no errors, %d warnings" % (res["counts"]["n_questions"], len(res["warnings"])))

bad = [dict(r) for r in rows[:6]]
bad[0]["source_location"] = "p.99"                                            # page that does not exist
bad[1]["source_location"] = "p.7"                                             # real page, but passage is on page 1
bad[2]["terminology_tags"] = "9"                                              # invalid category
bad[3]["assamese_question"] = bad[3]["english_question"]                      # not independently written
bad[4]["source_document"] = "MISSING-9"                                       # no cleaned text for this doc
bad[5]["terminology_tags"] = "6,2"                                            # tag 6 combined
r2 = validate(bad, clean, min_per_category=1)
joined = " | ".join(r2["errors"] + r2["warnings"])
for needle in ("p.99" if False else "not present", "IS on page", "invalid tag", "identical to the English", "no cleaned text file", "tag 6"):
    assert needle in joined, (needle, joined)
print("2b. validator catches: bad page, passage on a different page, bad tag, copied Assamese, missing doc, tag-6 misuse")

# ---------------------------------------------------------------- 3. dry run
out = os.path.join(td, "results")
rc = run_experiment.main(["--dataset", ds, "--clean-dir", clean, "--dry-run", "--out", out, "--seed", "7", "--k", "5"])
assert rc == 0
assert run_experiment.main(["--dataset", ds, "--clean-dir", clean, "--dry-run", "--out", out, "--seed", "7"]) == 2     # refuses to append
log = [json.loads(l) for l in open(os.path.join(out, "raw_log.jsonl"), encoding="utf-8")]
assert len(log) == 3 * N and log[0]["config"]["dry_run"] is True and len(log[0]["config"]["dataset_sha256"]) == 64
ann = list(csv.DictReader(open(os.path.join(out, "annotation.csv"), encoding="utf-8")))
key = {r["blind_id"]: r for r in csv.DictReader(open(os.path.join(out, "key.csv"), encoding="utf-8"))}
assert len(ann) == 3 * N and "system" not in ann[0]
bm = [r for r in log if r["system"] == "B" and r["question_id"] == "Q010"][0]
assert any(ch["doc_id"] == "DOC-001" and ch["page"] == "2" for ch in bm["retrieved"])             # BM25 finds the right page
print("3. dry run OK: %d log lines, blinded sheet has %d rows, retrieval hits the gold page" % (len(log), len(ann)))

# ---------------------------------------------------------------- 4. simulated raters, resolution, analysis
tags_of = {r["question_id"]: r["terminology_tags"] for r in rows}
def truth(bid):
    q, s = key[bid]["question_id"], key[bid]["system"]
    general = tags_of[q] == "6"
    if s == "A":
        return int(rng.choice([0, 1, 2], p=[.5, .3, .2])), int(rng.choice([0, 1, 2], p=[.6, .3, .1])), int(rng.random() < .5)
    c, g = int(rng.choice([0, 1, 2], p=[.04, .12, .84])), int(rng.choice([0, 1, 2], p=[.03, .10, .87]))
    if s == "C" and rng.random() < (0.10 if general else 0.50):
        c, g = max(0, c - 1), max(0, g - 1)
    return c, g, int(rng.random() < (0.10 if s == "B" else 0.18))
true = {b: truth(b) for b in key}
def write_rater(path, noise):
    cols = ["rater_correctness_0_2", "rater_groundedness_0_2", "rater_has_unsupported_claim_0_1"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ann[0].keys())); w.writeheader()
        for r in ann:
            v = list(true[r["blind_id"]])
            if noise and rng.random() < 0.10:
                j = int(rng.integers(0, 2)); v[j] = int(np.clip(v[j] + rng.choice([-1, 1]), 0, 2))
            r2 = dict(r); r2.update(dict(zip(cols, v))); w.writerow(r2)
ra, rb = os.path.join(td, "A.csv"), os.path.join(td, "B.csv")
write_rater(ra, False); write_rater(rb, True)
adir = os.path.join(out, "analysis")
assert analyze_results.main(["--make-resolved-template", "--rater-a", ra, "--rater-b", rb, "--out", adir]) == 0
tmpl = list(csv.DictReader(open(os.path.join(adir, "resolved_template.csv"), encoding="utf-8")))
n_blank = sum(1 for r in tmpl if r["disagreed_fields"])
assert 20 < n_blank < 200, n_blank
# an unresolved template must be refused, not silently analysed
try:
    analyze_results.main(["--log", os.path.join(out, "raw_log.jsonl"), "--key", os.path.join(out, "key.csv"), "--dataset", ds,
                          "--resolved", os.path.join(adir, "resolved_template.csv"), "--out", adir]); raise AssertionError("should refuse")
except SystemExit as e:
    assert "blanks" in str(e)
ta = analyze_results.read_scores(ra)
fm = {"correctness": "correctness", "groundedness": "groundedness", "unsupported_claim": "claim"}
with open(os.path.join(adir, "resolved.csv"), "w", newline="", encoding="utf-8") as f:       # team settles each blank (here: from the source = truth)
    w = csv.writer(f); w.writerow(["blind_id", "correctness", "groundedness", "unsupported_claim", "disagreed_fields"])
    for r in tmpl:
        w.writerow([r["blind_id"]] + [r[k] if r[k] != "" else ta[r["blind_id"]][v] for k, v in fm.items()] + [r["disagreed_fields"]])
assert analyze_results.main(["--log", os.path.join(out, "raw_log.jsonl"), "--key", os.path.join(out, "key.csv"), "--dataset", ds,
                             "--rater-a", ra, "--rater-b", rb, "--resolved", os.path.join(adir, "resolved.csv"), "--out", adir]) == 0
S = json.load(open(os.path.join(adir, "results_summary.json")))
md = open(os.path.join(adir, "results_report.md"), encoding="utf-8").read()
sy = S["systems"]
assert sy["B"]["mean_correctness"] > sy["A"]["mean_correctness"] + 0.8 and sy["B"]["mean_correctness"] > sy["C"]["mean_correctness"]
h1 = S["tests"]["H1_B_vs_C"][0]
assert h1["all"]["p_value"] < 0.001 and h1["all"]["mean_diff"] > 0 and h1["primary_only"]["p_value"] < 0.01
h2 = S["tests"]["H2_A_vs_B"][0]
assert h2["all"]["p_value"] < 0.001 and h2["all"]["mean_diff"] < 0                  # A - B is negative: B better
h3 = S["tests"]["H3_terminology_vs_general"]["correctness"]
assert h3["n_terminology"] == 100 and h3["n_general"] == 50 and h3["p_value"] < 0.05 and h3["rank_biserial"] > 0
assert 0.5 < S["kappa"]["correctness"]["kappa"] < 1.0
assert sy["B"]["recall_at_k"] > 0.95 and sy["B"]["n_with_gold_page"] == 120          # news questions excluded from retrieval scoring
assert "H1: System B vs System C" in md and "H3" in md
print("4. analysis OK: B-A correctness %.2f, B-C gap p=%.1e, H3 p=%.3f, kappa(correctness)=%.2f, recall@5(B)=%.2f" % (
    sy["B"]["mean_correctness"] - sy["A"]["mean_correctness"], h1["all"]["p_value"], h3["p_value"], S["kappa"]["correctness"]["kappa"], sy["B"]["recall_at_k"]))
print("\nALL TOOL TESTS PASSED")
