"""Run: python tests/test_pipeline.py   (uses stand-ins; needs no models, keys or internet)"""
import sys, os, csv, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chunking import chunk_text
from src.systems import system_a, system_b, system_c, ABSTAIN_TOKEN
from src.harness import run_all, export_blinded

# ---- chunking
text = " ".join(f"w{i}" for i in range(1000))
ch = chunk_text(text, "DOC1", "4", chunk_words=300, overlap=50)
assert ch[0]["text"].split()[0] == "w0" and len(ch[0]["text"].split()) == 300
assert ch[1]["text"].split()[0] == "w250"            # overlap of 50 words
assert ch[-1]["text"].split()[-1] == "w999"          # nothing dropped at the end
assert len({c["chunk_id"] for c in ch}) == len(ch)   # unique ids
print("chunking OK:", len(ch), "chunks")

# ---- stand-ins for the real models
calls = {"as_en": 0, "en_as": 0}
def fake_as_to_en(t): calls["as_en"] += 1; return "ENG(" + t + ")"
def fake_en_to_as(t): calls["en_as"] += 1; return "ASM(" + t + ")"
def fake_retrieve(q, k): return [{"chunk_id": f"c{i}", "doc_id": "D", "page": "1", "text": "ctx"} for i in range(k)]
def llm_answers(prompt): return "Rs 6 lakh."
def llm_abstains(prompt): return ABSTAIN_TOKEN

a = system_a("q", llm_answers)
assert a["retrieved"] == [] and a["abstained"] is False
b = system_b("q", fake_retrieve, llm_answers, k=3)
assert len(b["retrieved"]) == 3 and b["abstained"] is False
c = system_c("asm-q", fake_as_to_en, fake_en_to_as, fake_retrieve, llm_answers, k=3)
assert c["translated_question_en"] == "ENG(asm-q)" and c["final_answer"] == "ASM(Rs 6 lakh.)"
assert calls == {"as_en": 1, "en_as": 1}
# abstention must NOT be translated, and must be detected
c2 = system_c("asm-q", fake_as_to_en, fake_en_to_as, fake_retrieve, llm_abstains)
assert c2["abstained"] is True and c2["final_answer"] == ABSTAIN_TOKEN and calls["en_as"] == 1
print("systems A/B/C logic OK (incl. abstention passthrough)")

# ---- harness + blinding
dataset = [{"question_id": f"Q{i}", "english_question": f"EN{i}", "assamese_question": f"AS{i}",
            "supporting_passage": f"P{i}", "reference_answer": f"R{i}"} for i in range(1, 6)]
systems = {"A": lambda r: system_a(r["english_question"], llm_answers),
           "B": lambda r: system_b(r["english_question"], fake_retrieve, llm_answers),
           "C": lambda r: system_c(r["assamese_question"], fake_as_to_en, fake_en_to_as, fake_retrieve, llm_answers)}
with tempfile.TemporaryDirectory() as td:
    recs = run_all(dataset, systems, {"k": 5, "llm": "stub"}, os.path.join(td, "log.jsonl"))
    assert len(recs) == 15
    lines = open(os.path.join(td, "log.jsonl"), encoding="utf-8").read().strip().split("\n")
    assert len(lines) == 15 and json.loads(lines[0])["config"]["k"] == 5
    ann, key = os.path.join(td, "ann.csv"), os.path.join(td, "key.csv")
    export_blinded(recs, dataset, ann, key, seed=1)
    ann_rows = list(csv.DictReader(open(ann, encoding="utf-8")))
    key_rows = list(csv.DictReader(open(key, encoding="utf-8")))
    assert len(ann_rows) == len(key_rows) == 15
    assert "rater_has_unsupported_claim_0_1" in ann_rows[0] and "system" not in ann_rows[0] and "A" not in str(list(ann_rows[0].keys()))  # no system column leaks
    # System C rows must show the Assamese question; others the English one
    kmap = {k["blind_id"]: k for k in key_rows}
    for r in ann_rows:
        sysname = kmap[r["blind_id"]]["system"]
        assert r["question"].startswith("AS" if sysname == "C" else "EN")
    # shuffle is deterministic for a given seed
    export_blinded(recs, dataset, ann + "2", key + "2", seed=1)
    assert open(ann, encoding="utf-8").read() == open(ann + "2", encoding="utf-8").read()
print("harness logging + blinded export OK")

# ---- ingest (B -> A hand-off format)
from src.ingest import load_clean_doc, ingest_directory
with tempfile.TemporaryDirectory() as td:
    open(os.path.join(td, "WII-001.txt"), "w", encoding="utf-8").write(
        "front matter\n<<<PAGE 12>>>\nElephants died. " + "word " * 400 + "\n<<<PAGE 13>>>\nSecond page text.\n")
    open(os.path.join(td, "LS-001.txt"), "w", encoding="utf-8").write("<<<PAGE 3>>>\nParliament reply text.\n")
    pages = load_clean_doc(os.path.join(td, "WII-001.txt"))
    assert [p for p, _ in pages] == ["x", "12", "13"]
    chunks = ingest_directory(td, chunk_words=300, overlap=50)
    assert {c["doc_id"] for c in chunks} == {"WII-001", "LS-001"}
    assert any(c["doc_id"] == "WII-001" and c["page"] == "13" for c in chunks)
    assert all(c["chunk_id"].startswith(c["doc_id"] + "::p" + c["page"]) for c in chunks)
print("ingest hand-off format OK:", len(chunks), "chunks with doc_id + page preserved")
print("\nALL PIPELINE TESTS PASSED")
