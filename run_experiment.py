"""Run Systems A, B and C on the frozen dataset and produce the blinded annotation sheet.

  Rehearsal (no models, no keys, no internet - proves the wiring):
      python run_experiment.py --dataset data/questions.csv --clean-dir data/clean --dry-run --out results_dry
  Real run (in Colab, after translate.py / kb.py / the Groq key work):
      python run_experiment.py --dataset HWC_Project_Workbook.xlsx --clean-dir data/clean --llm-model <groq model> --k 5 --seed 4821

--seed: Person B should choose it privately. Then KEEP results/key.csv OUT of the shared folder and do not open it
until both raters have finished scoring.
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset_io import load_dataset, validate, format_report, file_sha256
from src.ingest import ingest_directory
from src.systems import system_a, system_b, system_c
from src.harness import run_all, export_blinded


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--out", default="results")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--chunk-words", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--kb-path", default="kb_store")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N questions (for smoke tests)")
    ap.add_argument("--force", action="store_true", help="run even if validation reports errors (NOT for the final run)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    rows = load_dataset(a.dataset)
    res = validate(rows, a.clean_dir)
    print(format_report(res))
    if res["errors"] and not a.force:
        print("\nStopping: fix the errors above (or use --force for a rehearsal).")
        return 2
    if a.limit:
        rows = rows[:a.limit]

    os.makedirs(a.out, exist_ok=True)
    log_path = os.path.join(a.out, "raw_log.jsonl")
    if os.path.exists(log_path):
        if not a.overwrite:
            print("Refusing to append to existing %s. Use --overwrite to replace it, or a new --out folder." % log_path)
            return 2
        os.remove(log_path)

    chunks = ingest_directory(a.clean_dir, a.chunk_words, a.overlap)
    print("\nKnowledge base: %d chunks from %s" % (len(chunks), a.clean_dir))

    if a.dry_run:
        from src.lexical import LexicalIndex
        from src.llm import dry_llm
        retrieve = LexicalIndex(chunks).retrieve
        llm = dry_llm
        oracle = {r["assamese_question"]: r["english_question"] for r in rows}      # cheats on purpose: plumbing test only
        as_to_en = lambda t: oracle.get(t, t)
        en_to_as = lambda t: "[AS] " + t
        model_name, embed_name = "DRY-RUN", "BM25-lexical"
    else:
        from src.kb import KnowledgeBase, EMBED_MODEL
        from src.translate import make_translators
        from src.llm import make_llm
        if not a.llm_model:
            print("--llm-model is required for a real run (copy the current name from the Groq console).")
            return 2
        kb = KnowledgeBase(path=a.kb_path)
        if kb.col.count() == 0:
            for i in range(0, len(chunks), 256):
                kb.add_chunks(chunks[i:i + 256])
        retrieve, llm = kb.retrieve, make_llm(a.llm_model, temperature=0.0)
        as_to_en, en_to_as = make_translators()
        model_name, embed_name = a.llm_model, EMBED_MODEL

    systems = {"A": lambda r: system_a(r["english_question"], llm),
               "B": lambda r: system_b(r["english_question"], retrieve, llm, a.k),
               "C": lambda r: system_c(r["assamese_question"], as_to_en, en_to_as, retrieve, llm, a.k)}
    config = {"k": a.k, "llm": model_name, "temperature": 0.0, "embedding": embed_name, "chunk_words": a.chunk_words,
              "overlap": a.overlap, "dry_run": a.dry_run, "dataset_file": os.path.basename(a.dataset),
              "dataset_sha256": file_sha256(a.dataset), "n_questions": len(rows), "n_chunks": len(chunks),
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(config, open(os.path.join(a.out, "run_config.json"), "w"), indent=2)

    t0 = time.time()
    records = run_all(rows, systems, config, log_path)
    print("Ran %d (question, system) pairs in %.1fs -> %s" % (len(records), time.time() - t0, log_path))
    ann, key = os.path.join(a.out, "annotation.csv"), os.path.join(a.out, "key.csv")
    export_blinded(records, rows, ann, key, seed=a.seed)
    print("Blinded sheet for the raters: %s\nKEY (move out of the shared folder; do not open until scoring is done): %s" % (ann, key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
