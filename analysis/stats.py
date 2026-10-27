"""
Statistics for the HWC multilingual RAG study (Protocol v1).

Everything here works on plain lists / pandas columns, so it can be tested
without any model, API key or GPU. Run this file directly to execute the
self-test:  python analysis/stats.py
"""
from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon, rankdata
from sklearn.metrics import cohen_kappa_score


# ---------------------------------------------------------------- agreement
def weighted_kappa(rater_a, rater_b, weights="linear"):
    """Cohen's weighted kappa for ordinal 0/1/2 scores.
    'linear' weighting: a 0-vs-2 disagreement costs twice a 0-vs-1 disagreement.
    Rough reading (Landis & Koch): <0.2 poor, .2-.4 fair, .4-.6 moderate,
    .6-.8 substantial, >.8 almost perfect."""
    return cohen_kappa_score(rater_a, rater_b, weights=weights, labels=[0, 1, 2])


# ------------------------------------------------------- paired significance
def paired_wilcoxon(scores_x, scores_y):
    """Wilcoxon signed-rank test on PAIRED scores (same question, two systems).
    Returns dict with statistic, p-value, median difference, and an effect size
    (matched-pairs rank-biserial correlation, range -1..1).
    Questions where both systems got the same score carry no information and
    are dropped by the test, so we report how many were dropped."""
    x = np.asarray(scores_x, dtype=float)
    y = np.asarray(scores_y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Scores must be paired: same questions, same order.")
    d = x - y
    n_tied = int(np.sum(d == 0))
    n_used = int(np.sum(d != 0))
    if n_used == 0:
        return dict(n=len(d), n_tied=n_tied, n_used=0, statistic=None, p_value=1.0,
                    median_diff=0.0, mean_diff=0.0, rank_biserial=0.0)
    res = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    # rank-biserial effect size from signed ranks
    nz = d[d != 0]
    ranks = rankdata(np.abs(nz))
    r_plus = ranks[nz > 0].sum()
    r_minus = ranks[nz < 0].sum()
    rb = (r_plus - r_minus) / (r_plus + r_minus)
    return dict(n=len(d), n_tied=n_tied, n_used=n_used, statistic=float(res.statistic),
                p_value=float(res.pvalue), median_diff=float(np.median(d)),
                mean_diff=float(np.mean(d)), rank_biserial=float(rb))



# ------------------------------------------- news-sourced vs primary check
def sensitivity_wilcoxon(df, col_x, col_y, grade_col="evidence_grade"):
    """Run the same paired test on (a) ALL questions and (b) PRIMARY-sourced questions only.
    If both tell the same story, news-sourced questions did not drive the conclusion.
    df needs one row per question, with score columns for the two systems being compared."""
    out = {"all": paired_wilcoxon(df[col_x], df[col_y])}
    prim = df[df[grade_col] == "primary"]
    out["n_primary"] = int(len(prim))
    out["primary_only"] = paired_wilcoxon(prim[col_x], prim[col_y]) if len(prim) > 0 else None
    return out

# --------------------------------------------------------- retrieval metrics
def recall_at_k(retrieved_ids_per_q, relevant_ids_per_q, k=5):
    """Fraction of questions where at least one correct source chunk is in the top-k."""
    hits = 0
    for retrieved, relevant in zip(retrieved_ids_per_q, relevant_ids_per_q):
        if set(retrieved[:k]) & set(relevant):
            hits += 1
    return hits / len(retrieved_ids_per_q)


def mean_reciprocal_rank(retrieved_ids_per_q, relevant_ids_per_q):
    """Average of 1/rank of the first correct chunk (0 if none retrieved)."""
    rr = []
    for retrieved, relevant in zip(retrieved_ids_per_q, relevant_ids_per_q):
        score = 0.0
        for rank, cid in enumerate(retrieved, start=1):
            if cid in set(relevant):
                score = 1.0 / rank
                break
        rr.append(score)
    return float(np.mean(rr))


# ------------------------------------------------------- answer-level rates
def hallucination_rate(groundedness_scores, threshold=0):
    """Share of answers judged NOT supported by retrieved evidence
    (groundedness <= threshold; default 0 = unsupported)."""
    g = np.asarray(groundedness_scores)
    return float(np.mean(g <= threshold))


def false_abstention_rate(abstained_flags, answerable_flags):
    """Among questions that WERE answerable from the evidence, how often did the
    system say 'insufficient information' anyway?"""
    a = np.asarray(abstained_flags, dtype=bool)
    ans = np.asarray(answerable_flags, dtype=bool)
    if ans.sum() == 0:
        return 0.0
    return float(np.sum(a & ans) / ans.sum())


# ------------------------------------------------------ H3 subgroup helper
def subgroup_means(df, score_col, tag_col, tag_values=(1, 2, 3, 4, 5, 6)):
    """Mean score per terminology category. A question with tags '2,4' is counted
    in BOTH category 2 and 4. df[tag_col] holds strings like '2,4'."""
    out = {}
    for t in tag_values:
        mask = df[tag_col].astype(str).apply(lambda s: str(t) in [x.strip() for x in s.split(",")])
        out[t] = (float(df.loc[mask, score_col].mean()) if mask.any() else None, int(mask.sum()))
    return out



# ------------------------------------------------ confidence interval (Protocol s13)
def bootstrap_ci_mean_diff(scores_x, scores_y, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the MEAN paired difference (x - y).
    Resample questions with replacement n_boot times; the middle (1-alpha) of the resampled
    means is the interval. Protocol s13 asks for effect sizes AND CIs, not just p-values."""
    d = np.asarray(scores_x, dtype=float) - np.asarray(scores_y, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


# ----------------------------------------------------- H3 (Protocol s3, s13)
def _tagset(x):
    return {t.strip() for t in str(x).split(",") if t.strip()}


def h3_degradation_test(df, score_b, score_c, tags_col, general_tag="6", alternative="two-sided"):
    """H3: do questions with HWC terminology lose MORE from translation than general-language ones?
    Degradation per question = score under System B (English) minus score under System C (Assamese).
    Group T = questions with at least one tag in 1-5.  Group G = questions tagged ONLY '6' (general).
    Compared with a Mann-Whitney U test (two independent groups, ordinal-friendly).
    Effect size = rank-biserial correlation (-1..1; positive = T degrades more than G).
    The grouping rule and 'alternative' must be FROZEN before outcomes are seen (Protocol s13)."""
    deg = df[score_b].astype(float) - df[score_c].astype(float)
    sets = df[tags_col].apply(_tagset)
    in_g = sets.apply(lambda t: t == {general_tag})
    in_t = sets.apply(lambda t: len(t - {general_tag}) > 0)
    dt, dg = deg[in_t].to_numpy(), deg[in_g].to_numpy()
    if len(dt) == 0 or len(dg) == 0:
        return dict(n_terminology=int(len(dt)), n_general=int(len(dg)), p_value=None, note="one group is empty")
    from scipy.stats import mannwhitneyu
    res = mannwhitneyu(dt, dg, alternative=alternative)
    rb = 2 * res.statistic / (len(dt) * len(dg)) - 1
    return dict(n_terminology=int(len(dt)), n_general=int(len(dg)),
                mean_degradation_terminology=float(dt.mean()), mean_degradation_general=float(dg.mean()),
                U=float(res.statistic), p_value=float(res.pvalue), rank_biserial=float(rb))


# --------------------------------------- hallucination, claim level (Protocol s10.4)
def unsupported_claim_rate(flags):
    """Protocol s10.4: share of answers containing at least one substantive unsupported claim.
    flags = one 0/1 per answer from the rater (1 = has an unsupported claim). This is DIFFERENT from
    hallucination_rate() above, which only counts groundedness==0 answers (the s10.3 unsupported-answer rate)."""
    return float(np.mean(np.asarray(flags, dtype=float)))

# =========================================================== SELF-TEST
if __name__ == "__main__":
    import pandas as pd
    rng = np.random.default_rng(42)

    # 1. kappa: identical raters -> 1.0 ; independent random raters -> near 0
    same = rng.integers(0, 3, 200)
    assert abs(weighted_kappa(same, same) - 1.0) < 1e-9
    k_rand = weighted_kappa(rng.integers(0, 3, 400), rng.integers(0, 3, 400))
    assert abs(k_rand) < 0.15, k_rand
    # known worked example: perfect agreement except one 0-vs-2 swap
    print("kappa (identical):", weighted_kappa(same, same))
    print("kappa (random raters, expect ~0):", round(k_rand, 3))

    # 2. Wilcoxon: build a System B that is clearly better than System C
    n = 180
    sysB = rng.choice([0, 1, 2], size=n, p=[0.10, 0.30, 0.60])
    drop = rng.random(n) < 0.35          # in 35% of questions C is one point worse
    sysC = np.clip(sysB - drop.astype(int), 0, 2)
    r = paired_wilcoxon(sysB, sysC)
    print("Wilcoxon B vs C (expect small p, B better):", r)
    assert r["p_value"] < 0.001 and r["median_diff"] >= 0
    # identical systems -> p = 1
    r0 = paired_wilcoxon(sysB, sysB)
    assert r0["p_value"] == 1.0 and r0["n_used"] == 0

    # 3. retrieval metrics on a tiny hand-checked example
    retrieved = [["c1", "c2", "c3"], ["c9", "c4", "c5"], ["c7", "c8", "c6"]]
    relevant = [["c1"], ["c4"], ["c0"]]
    rec = recall_at_k(retrieved, relevant, k=3)       # hits: q1 yes, q2 yes, q3 no -> 2/3
    mrr = mean_reciprocal_rank(retrieved, relevant)   # (1 + 1/2 + 0)/3 = 0.5
    print("recall@3 =", round(rec, 3), "(expected 0.667) | MRR =", round(mrr, 3), "(expected 0.5)")
    assert abs(rec - 2 / 3) < 1e-9 and abs(mrr - 0.5) < 1e-9

    # 4. answer-level rates
    assert abs(hallucination_rate([0, 1, 2, 0, 2]) - 0.4) < 1e-9
    fa = false_abstention_rate([1, 1, 0, 0], [1, 0, 1, 1])  # answerable: q1,q3,q4 ; abstained&answerable: q1 -> 1/3
    assert abs(fa - 1 / 3) < 1e-9
    print("hallucination rate / false-abstention checks passed")

    # 5. H3 subgroup helper, incl. multi-tag questions
    df = pd.DataFrame({"tags": ["2,4", "4", "2", "1,6"], "score": [2, 1, 0, 2]})
    sm = subgroup_means(df, "score", "tags")
    assert sm[4] == (1.5, 2) and sm[2] == (1.0, 2) and sm[3] == (None, 0)
    print("subgroup means:", sm)

    # 6. all-vs-primary sensitivity check
    d = pd.DataFrame({"B": sysB, "C": sysC, "evidence_grade": rng.choice(["primary", "news-2plus"], size=len(sysB), p=[0.7, 0.3])})
    sens = sensitivity_wilcoxon(d, "B", "C")
    assert sens["n_primary"] == int((d["evidence_grade"] == "primary").sum())
    assert sens["all"]["n"] == len(d) and sens["primary_only"]["n"] == sens["n_primary"]
    print("sensitivity: all p =", round(sens["all"]["p_value"], 6), "| primary-only p =", round(sens["primary_only"]["p_value"], 6), "| n_primary =", sens["n_primary"])
    d0 = pd.DataFrame({"B": [1, 2], "C": [1, 2], "evidence_grade": ["news-1", "news-1"]})
    assert sensitivity_wilcoxon(d0, "B", "C")["primary_only"] is None   # no primary questions -> handled, not a crash

    # 7. bootstrap CI: constant difference -> zero-width interval; noisy -> sensible width containing truth
    m, lo, hi = bootstrap_ci_mean_diff([3, 3, 3, 3], [2, 2, 2, 2], n_boot=500)
    assert (m, lo, hi) == (1.0, 1.0, 1.0)
    base = rng.normal(0.3, 0.6, 180)                      # true mean difference 0.3
    m, lo, hi = bootstrap_ci_mean_diff(base, np.zeros(180), n_boot=4000, seed=1)
    assert lo < 0.3 < hi and 0.10 < (hi - lo) < 0.25, (lo, hi)
    print("bootstrap CI ok: mean %.3f  95%% CI [%.3f, %.3f]" % (m, lo, hi))

    # 8. H3: build data where terminology questions clearly degrade more than general ones
    n_t, n_g = 110, 70
    B = rng.choice([1, 2], size=n_t + n_g, p=[0.2, 0.8])
    worse = np.concatenate([rng.random(n_t) < 0.55, rng.random(n_g) < 0.10])
    C = np.clip(B - worse.astype(int), 0, 2)
    tags = ["2,4"] * 60 + ["1"] * 50 + ["6"] * 70
    dh = pd.DataFrame({"B": B, "C": C, "tags": tags})
    h3 = h3_degradation_test(dh, "B", "C", "tags")
    print("H3:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in h3.items()})
    assert h3["n_terminology"] == 110 and h3["n_general"] == 70
    assert h3["p_value"] < 0.001 and h3["rank_biserial"] > 0.3
    # a question tagged '6,2' counts as terminology (it has a non-general tag), not general
    dm = pd.DataFrame({"B": [2, 2, 2], "C": [1, 2, 2], "tags": ["6,2", "6", "6"]})
    hm = h3_degradation_test(dm, "B", "C", "tags")
    assert hm["n_terminology"] == 1 and hm["n_general"] == 2
    # empty group handled, not a crash
    assert h3_degradation_test(pd.DataFrame({"B": [1], "C": [1], "tags": ["6"]}), "B", "C", "tags")["p_value"] is None

    # 9. claim-level unsupported rate
    assert abs(unsupported_claim_rate([1, 0, 0, 1, 0]) - 0.4) < 1e-9
    print("\nALL SELF-TESTS PASSED")
