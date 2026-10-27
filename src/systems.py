"""The three systems from Protocol Section 4, written as plain functions.
The translator, retriever and LLM are PASSED IN, so the logic can be tested with stand-ins
(see tests/test_pipeline.py) and swapped for the real thing in Colab.

 A: question -> LLM                                   (no retrieval)
 B: English question -> retrieve -> LLM               (English RAG)
 C: Assamese question -> English -> B -> Assamese     (translation wrapped around B)
"""
from __future__ import annotations

ABSTAIN_TOKEN = "INSUFFICIENT_INFORMATION"

PROMPT_A = "Answer the question as accurately as you can.\n\nQuestion: {q}\nAnswer:"

PROMPT_RAG = (
    "You are answering questions about human-wildlife conflict in Assam using ONLY the context below.\n"
    "Rules: (1) Use only facts stated in the context. (2) If the context does not contain enough "
    f"information to answer, reply with exactly {ABSTAIN_TOKEN}. (3) Do not add outside knowledge.\n\n"
    "Context:\n{ctx}\n\nQuestion: {q}\nAnswer:"
)


def format_context(chunks):
    return "\n\n".join(f"[{c['doc_id']} p.{c.get('page','')}] {c['text']}" for c in chunks)


def system_a(question_en, llm):
    ans = llm(PROMPT_A.format(q=question_en))
    return {"final_answer": ans, "english_answer": ans, "retrieved": [], "abstained": False}


def system_b(question_en, retrieve, llm, k=5):
    chunks = retrieve(question_en, k)
    ans = llm(PROMPT_RAG.format(ctx=format_context(chunks), q=question_en))
    return {"final_answer": ans, "english_answer": ans, "retrieved": chunks,
            "abstained": ABSTAIN_TOKEN in ans}


def system_c(question_as, as_to_en, en_to_as, retrieve, llm, k=5):
    q_en = as_to_en(question_as)                       # translation step 1
    inner = system_b(q_en, retrieve, llm, k)           # exactly System B on the translated question
    # Do not translate the abstain token into Assamese: keep it detectable and translate only real answers.
    final = inner["english_answer"] if inner["abstained"] else en_to_as(inner["english_answer"])
    return {"final_answer": final, "english_answer": inner["english_answer"], "retrieved": inner["retrieved"],
            "abstained": inner["abstained"], "translated_question_en": q_en}
