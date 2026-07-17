# Notebook 15: How Accurate Are the Machine Labelers Against Human Judgment?

*Do the embedding (AIME) and Haiku file labelers reproduce human category judgments — and does it matter which one the maturity pipeline uses?*

**Dataset**: 195 strict-whitelisted AI artifacts from the November 2025 snapshot, drawn from 35 repositories (out of 268 eligible; 509 in the msrc corpus, 392 with any detected artifact) in seven level-balanced blocks of five — one repository of every Haiku-derived maturity level L1–L4 per block plus one extra. Seeded, reproducible draw (notebook 14, seed 14).
**Labels collected**: 270 independent human labels from three raters under an overlap design — 128 items with one vote, 59 with two, 8 with three. Forced choice among 11 options (9 leveled artifact categories + `general-documentation` + `none`), fully blinded to machine labels. All three returned packs were complete and valid (92 / 86 / 92 rows). Evaluation collapses `general-documentation` and `none` into `not-artifact` (10 classes), mirroring notebook 13.
**Reference labels**: majority vote where an item has ≥2 votes, the single rater's label on unique repos; 20 items with no majority are excluded → **175 reference items** (47 majority-backed, 128 single-rater).
**Data lineage**: machine labels come from `data/validation/human_labeling/manifest.csv`, verified in-notebook against `msrc_file_predictions{,_llm}.parquet` (staleness guard). Raw returned workbooks live untouched in `human_labeling/labeled results/`; normalized canonical copies in `human_labeling/returned/`. All numbers below are computed by `15. human_label_evaluation.ipynb`; the executed run is frozen as **`15. human_label_evaluation.html`** and every statistic is exported to `human_labeling/paper/summary.json` plus five LaTeX tables (`paper/table_*.tex`).

> **Comparison caveat.** The 35-repo sample is stratified on the *Haiku-derived* maturity level, deliberately over-representing the rare L1/L3/L4 strata (population: 15×L1 / 222×L2 / 24×L3 / 7×L4). Every repo-level rate in this document is therefore **stratum-weighted, not a population rate** — the full 35-repo table is always reported alongside.

---

## Statistical Methods

### Krippendorff's α (headline inter-annotator agreement)

**What it measures.** Chance-corrected agreement among any number of raters, tolerant of missing data — each item may carry a different number of votes.

**Why we use it.** The overlap design yields 1–3 votes per item. Fleiss' κ requires a constant rater count (here only the 8 triple-labeled items qualify); Cohen's κ handles exactly two raters. α is the only statistic that uses *all* 67 multiply-labeled items, so it is the citable agreement figure.

**How to interpret.** α ≥ 0.80 is conventionally "reliable", 0.67–0.80 "tentative", below 0.67 weak. Our **α = 0.572** (identical on the raw 11-option labels — no multiply-voted item split between `general-documentation` and `none`) means human agreement on this task is *moderate*: the categories are genuinely hard to separate at the margins (`rules` vs `code-style` vs `skills`; `commands` vs `flows`). This is the ceiling context against which machine accuracy must be read.

### Cohen's κ and Fleiss' κ

Pairwise (Cohen) and fixed-panel (Fleiss) chance-corrected agreement. Reported as secondary IAA statistics — Fleiss' κ = 0.839 looks excellent but rests on **8 items** and should not be leaned on. Cohen's κ is also used to compare each *machine* against the human reference: unlike raw accuracy, it discounts the agreement expected from the skewed class distribution (73 of 175 references are `commands`, 55 are `rules`).

### Wilson score interval

95% CI for a binomial proportion; preferred over the normal approximation at moderate n and extreme proportions. Used for all accuracy and match rates.

### Repo-cluster bootstrap

Items are not independent — they cluster within 35 repositories (one repo contributes up to 40 items with correlated content and difficulty). Item-i.i.d. Wilson CIs are therefore anti-conservative. We resample the 35 repos with replacement (10,000 draws, seed 15) and recompute each statistic; the percentile interval is the honest uncertainty at the file level, and the plain repo bootstrap is the right scheme at the repo level (the repo is the sampling unit).

### Exact McNemar test

Paired comparison of the two machine labelers on the same items: of the items where exactly one labeler is right, is the split compatible with 50/50? Exact binomial version, appropriate at small discordant counts (here 13 vs 12).

### Balanced accuracy and macro F1

Raw accuracy is dominated by the two big classes. Balanced accuracy averages per-class recall; macro F1 averages per-class F1 over the 9 classes with human support > 0. Both expose failure on rare categories that raw accuracy hides.

### Quadratic weighted κ (repo level)

Maturity levels are ordinal; a two-level miss is worse than a one-level miss. QWK penalizes disagreements by squared distance, complementing the exact-match rate.

---

## Findings

### F1. Human agreement is moderate — the task itself is hard

Krippendorff's α = 0.572 over the 67 multiply-labeled items; pairwise Cohen's κ = 0.649 (A–B, 76.0% raw), 0.499 (A–C, 64.1%), 0.812 (B–C, 89.5%). Rater C used only 5 of the 11 options (45× `rules`, 37× `commands`, never `general-documentation`/`none`), which depresses the A–C pair. 20 of 195 items produced no majority and were dropped. Any machine-accuracy claim below must be read against this ceiling: **humans themselves agree only 64–90% of the time pairwise.**

### F2. At the file level the two machine labelers are statistically indistinguishable — both ≈ 82% against the human reference

| metric (n = 175) | embedding (AIME) | Haiku |
|---|---|---|
| accuracy (Wilson 95% CI) | **81.7%** (75.3–86.7) | **81.1%** (74.7–86.2) |
| accuracy (repo-cluster bootstrap) | 63.3–95.6 | 67.5–90.3 |
| Cohen's κ vs reference | 0.743 | 0.739 |
| balanced accuracy | 0.514 | 0.548 |
| macro F1 | 0.480 | 0.525 |
| weighted F1 | 0.808 | 0.823 |

McNemar exact p = 1.000 (13 vs 12 discordant items); bootstrap 95% CI on the difference −6.7 to +9.4 pp. Substantial κ (≈ 0.74) for both — the ~82% is not an artifact of class skew. Note the honest cluster-bootstrap intervals are wide (±15 pp): 35 repos is enough to compare labelers, not to certify a third digit of accuracy.

### F3. Both labelers succeed on the same categories and fail on the same categories

Strong for both: `commands` (F1 0.944 emb / 0.972 Haiku, n=73), `rules` (0.862 / 0.822, n=55), `configuration` (1.0 / 1.0, n=5), `agents` (0.774 / 0.774 — perfect precision, recall 0.632, n=19). Broken for both: `skills` (F1 = 0, n=5), `flows` (F1 = 0, n=3), `architecture` (0, n=1); `code-style` favors Haiku (0.571 vs 0.286, n=6). The 20 both-machines-wrong items are systematic, not random: humans read `.cursor/plans/*.plan.md` as `flows`, `.github/workflows/*` agentic configs as `agents`, and procedure-like `.cursor/rules/*.mdc` as `skills`/`code-style`, while the machines fall back to `rules` or `not-artifact`. `not-artifact` precision is low for both (0.296 / 0.438): the machines discard artifacts humans consider real more often than the reverse.

### F4. At the repository level the embedding labeler tracks human judgment better than Haiku

Propagated through the production `aggregate_repo_maturity()` on the identical file subset:

| metric (35 repos) | embedding (AIME) | Haiku |
|---|---|---|
| exact level match | **34/35 = 97%** (Wilson 85–99; bootstrap 91–100) | **27/35 = 77%** (61–88; 63–91) |
| within ±1 level | 97% | 97% |
| quadratic weighted κ | 0.829 | 0.737 |

Haiku's eight misses are directional, not noise: it **under-rates five of the seven L1-stratum repos** (assigns L1 where the human-derived level is L2 — single-artifact repos where Haiku labels the file `not-artifact`/unleveled) and **over-rates three L4-stratum repos** (`phel-lang` L4 vs human L2, `Z3` and `mastra` L4 vs L3 — `flows`/`session-logs` calls the raters do not corroborate). Of the four repos Haiku placed at L4 within the evaluated subset, humans confirm **one** (`brooksy4503/chatlima`). The embedding labeler's single miss is the mirror image and is structural: it assigns no `flows`/`session-logs` labels at all, so it *cannot* produce L4 and scores `chatlima` L2 against a human L4.

**Implication for the pipeline**: embedding labels are the safer default for repo-level maturity (97% human agreement, κ_w 0.83), with the caveat that true L4 repos — rare in the population (7 of 268 by Haiku's own count, and Haiku over-counts) — are invisible to it. Haiku's L4 assignments specifically should be treated as low-precision flags, not levels.

### F5. Results are robust to how the human reference is constructed

- **Reference strength.** Majority-vote references (n=47): embedding 93.6%, Haiku 78.7%. Single-rater references (n=128): 77.3% / 82.0%. The embedding labeler looks *better* exactly where the human signal is strongest; neither ordering reverses the headline conclusion (parity at file level).
- **Each rater as truth** (bypassing reference construction): embedding 67.4 / 76.7 / 84.8% against raters A/B/C; Haiku 69.6 / 73.3 / 75.0%. Machine–human agreement sits inside the human–human agreement range (64–90%) for every rater.
- **Confidence as a sanity check** (auxiliary): where rater A self-reported High confidence, machines agree with A 85–92%; at Low confidence, 25–33%. Machine disagreement concentrates precisely where the human found the item hard — consistent with moderate α rather than machine failure.

### Bottom line

Where notebook 13 established that the two machine labelers are *consistent with each other*, this study licenses the accuracy claim: **file-level agreement with human judgment is ≈ 82% (κ ≈ 0.74) for both labelers — indistinguishable from each other and comparable to human–human agreement on the same task — and the embedding labels reproduce the human-derived repository maturity level in 97% of sampled repositories (Haiku: 77%)**. The maturity pipeline's repo-level claims can rest on the embedding labeler; its known blind spot is the (rare, L4-defining) `flows`/`session-logs` categories, where neither labeler currently matches human perception.

---

## Limitations

1. **Stratum weighting.** The level-balanced draw over-represents L1/L3/L4; the 97%/77% repo rates are not population rates. (Weighted toward the 83%-L2 population, both labelers would look better.)
2. **Single-rater references.** 128 of 175 references rest on one rater. Mitigated by the reference-strength split (F5) — the embedding advantage *grows* on majority-backed items — and by the rater-as-truth bounds.
3. **Moderate human agreement.** α = 0.572 caps how sharply "accuracy" can be interpreted; part of the residual machine "error" is irreducible label ambiguity.
4. **Low support on rare categories.** `architecture` (n=1), `flows` (n=3), `session-logs` (n=0) metrics are anecdotal; `skills` (n=5) and `code-style` (n=6) are barely better. The F1 = 0 findings are directionally solid (all misses agree in kind) but the magnitudes are not estimates.
5. **Single snapshot.** November 2025 only, one content version per path — no temporal generalization.
6. **Rater C's narrow usage** (5 of 11 categories) depresses A–C agreement and colors 42 single-rater-C references; the majority mechanism bounds but does not remove this.
7. **Fleiss' κ = 0.839 rests on 8 items** — cite α, not Fleiss.

---

## Artifacts

| artifact | contents |
|---|---|
| `notebooks/15. human_label_evaluation.ipynb` | all computation (this document interprets, the notebook computes) |
| `notebooks/15. human_label_evaluation.html` | frozen executed run (immutable results) |
| `data/validation/human_labeling/paper/summary.json` | every statistic, machine-readable |
| `data/validation/human_labeling/paper/table_*.tex` | LaTeX tables: agreement, file-level, per-category, repo-level, full 35-repo |
| `data/validation/human_labeling/human_labels_reference.csv` | per-item reference labels + provenance |
| `data/validation/human_labeling/human_file_eval.csv` | per-category file-level metrics |
| `data/validation/human_labeling/human_repo_eval.csv` | 35-repo three-way maturity table |
| `data/validation/figures/notebook15_confusion_{embedding,haiku}.png` | confusion matrices vs human reference |
| `data/validation/figures/notebook15_repo_levels.png` | repo maturity, three label sources |
| `data/validation/human_labeling/labeled results/` | raw returned workbooks (originals, untouched) |
