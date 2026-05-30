# Does steering-vector geometry predict behaviour composition? An exact dyadic significance test

*A self-contained methods + results narrative for the paper. Covers (1) why our first
significance test was invalid, (2) what MRQAP / Mantel is and why it is the right tool,
(3) exactly what data goes in, and (4) how to read every figure. Companion to the
machine-generated numbers in
[`analysis/rq1_consolidated/mrqap_significance_results.md`](../analysis/rq1_consolidated/mrqap_significance_results.md).*

---

## 0. The question, and the one-sentence answer

We study **steering vectors**: directions in the residual stream of Llama-3.1-8B-Instruct
(layer 17) that, when added to the model's activations, push it toward a behaviour — `evil`,
`sycophantic`, `humorous`, and so on. Each behaviour has its own unit direction `v̂`.

When we inject **two** behaviours at once, they *compose*. Our central RQ1 hypothesis is that
the **geometry** of the two directions predicts the composition — concretely, that the
**signed cosine** between `v̂_i` and `v̂_j` predicts whether the two behaviours **reinforce**
each other (joint expression exceeds solo) or **suppress** each other (joint expression falls
short of solo). Aligned directions (cos > 0) should reinforce; opposed directions (cos < 0)
should suppress.

**One-sentence answer.** The directional effect is real but *more modest than our first
analysis claimed.* Under the primary dose-controlled scheme it is significant (standardized
partial slope **β = +0.54, exact p = 0.009**); under the stricter mechanical-null scheme it is
**borderline (β = +0.52, p = 0.051)**. Our earlier pooled regression reported p ≈ 0.0005–0.003;
that number was spuriously small because the test treated statistically **dependent** data as
if it were independent. This document explains the error, the fix, the data, and the figures.

---

## 1. Why the first approach was wrong

### 1.1 What we did the first time

The original test (in `rq1_v2v3_decomposition.ipynb`) pooled every `pair × scheme` row into
one long table — **n = 76** rows across three schemes, or **n = 50** across the two
dose-controlled ones — and fit an ordinary least-squares regression of the composition
outcome on the cosine (plus controls). To get a p-value it ran a **permutation test that
shuffled the cosine label across pairs, within each scheme**, ~10,000 times.

That regression and that permutation test both assume the rows are **independent
observations**. They are not. This single assumption is wrong in three compounding ways.

### 1.2 Problem A — behaviour pairs are not independent (dyadic dependence)

We have **8 behaviours** but **28 pairs** (every unordered pair, `C(8,2) = 28`). Each behaviour
appears in **7 different pairs**. So the row for `(evil, humorous)` and the row for
`(evil, sycophantic)` **share the `evil` steering vector**, its individual steering strength,
its judge-scoring idiosyncrasies — everything about `evil`. The 28 rows are built from only 8
underlying objects, so they are heavily **correlated**.

> **Analogy.** Imagine testing "do tall parents have tall children" but entering each family
> once per *sibling pair*. A family with 5 children contributes 10 pair-rows that all share the
> same parents. Treating those 10 rows as 10 independent data points makes your sample look 10×
> bigger — and your p-values 10× more confident — than the data warrant. Behaviour pairs are
> exactly this: the "siblings" are the 7 pairs each behaviour belongs to, and the shared
> "parent" is the behaviour's steering vector.

This is the defining feature of **dyadic / network data**: the rows are *edges*, but the
independent units are *nodes*. A test that randomizes edges as if they were independent
gets the effective sample size badly wrong.

### 1.3 Problem B — the cosine is double-counted across schemes

The cosine is a property of the two **directions**, not of the injection scheme. We verified
this directly: the cosine is **identical** across schemes (max difference = 0.0). So when we
pool `pair × scheme` rows, the *same cosine value* appears 2–3 times (once per scheme). The
pooled regression and the within-scheme shuffle treat those as separate, fresh observations of
the predictor. They are the same number. This inflates the apparent evidence a second time.

### 1.4 Problem C — the permutation null was built on the wrong unit

Shuffling the **cosine label across pairs** asks: "if I reassign cosine values to pairs at
random, how often do I see an association this strong?" But because of Problem A, a *pair* is
not an exchangeable unit. Reassigning `(evil, humorous)`'s cosine to some unrelated pair while
`evil`'s other six pairs keep their values produces a structure that **could never arise from
actually relabelling behaviours** — it breaks the dyadic constraint that one behaviour has one
vector and therefore one consistent set of cosines. The resulting null distribution is **too
narrow**, so the observed statistic looks more extreme than it should, and **p comes out too
small.** This is the "borrowed confidence" we needed to remove.

### 1.5 The consequence

The honest amount of information here is bounded by **8 behaviours / 28 pairs**, not 76 rows.
The first test effectively claimed ~76 independent observations and produced p ≈ 0.0005–0.003.
The correct test, below, treats the **8 behaviours** as the units that get permuted, and the
significance drops to the p = 0.009 / 0.051 range. *The effect did not disappear — it shrank to
its true size.*

---

## 2. The fix — Mantel and MRQAP, explained from scratch

The right family of tests for "is one dyadic matrix associated with another, respecting node
dependence?" is the **Quadratic Assignment Procedure (QAP)**: the **Mantel test** (one
predictor) and its regression generalization **MRQAP** (multiple predictors / controls).

### 2.1 Reframe the data as matrices

Stop thinking in rows. For a given scheme, arrange each pairwise quantity as a **symmetric
8 × 8 matrix** indexed by behaviour:

- `COS[i, j]` — signed cosine between `v̂_i` and `v̂_j` (the predictor).
- `Smat[i, j]` — the directional composition outcome for the pair (the main outcome).
- `Mmat[i, j]` — the joint-expression magnitude (a second outcome).
- `STR[i, j]`  — the pair's individual steering strength (a control).

The diagonal is undefined (a behaviour with itself) and ignored. The information lives in the
**upper triangle**: 28 cells (or fewer, after the coherence gate — see §3.6). The whole test
operates on these upper-triangle vectors.

### 2.2 The Mantel test (bivariate)

**Question:** are two matrices — say `COS` and `Smat` — associated beyond chance?

1. Flatten each matrix's upper triangle to a vector and compute the **correlation** between the
   two vectors. Call it the *Mantel r* (the observed association).
2. Now build the null **without** assuming the 28 cells are independent. Pick a permutation
   **π of the 8 behaviours** and relabel one matrix's **rows and columns simultaneously**:
   `COS[π(i), π(j)]`. This is the crucial move.
3. Recompute the correlation under that relabelling. Repeat over permutations → a **null
   distribution** of correlations.
4. **p = fraction of permutations whose |r| ≥ |observed r|** (two-sided).

### 2.3 Why permute *nodes*, not *edges*

Relabelling rows-and-columns together is a **relabelling of the 8 behaviours**. It keeps every
behaviour appearing in exactly 7 cells (the dyadic structure is preserved) and only scrambles
*which behaviour is which*. So it asks the correct counterfactual: *"if the cosines had been
attached to a different assignment of behaviours, how often would the COS↔Smat correspondence
be this strong?"* It respects Problem A by construction — the permuted matrix is always a
matrix that a real relabelling of behaviours could have produced. Edge-shuffling (the old test)
cannot make that guarantee, which is exactly why it was too liberal.

### 2.4 Exact, not sampled

With 8 behaviours there are only **8! = 40,320** permutations. That is small enough to
**enumerate every one of them**. So the p-value is **exact** — there is no Monte-Carlo
sampling error, no "we drew 10,000 random shuffles." Every reported p is a clean rational
fraction `count / 40,320`. (The smallest achievable p is `1/40,320 ≈ 2.5e-5`, because the
identity permutation always matches the observed statistic.)

### 2.5 MRQAP — adding a control (Dekker double semi-partialling)

Mantel is bivariate. But some behaviour pairs just steer **strongly** (e.g. `humorous`,
`sycophantic`) and others weakly (`formality`), regardless of geometry. A pair of two strong
behaviours will have large joint effects for reasons that have nothing to do with cosine. We
must control for this **individual strength** (`STR`). MRQAP is multiple regression on matrices:

- Regress the predictor `COS` on the control `STR` (as edge vectors) and keep the **residual**
  `rx` — the part of cosine *not* explained by strength.
- Regress the outcome `Smat` on `STR` and keep its residual `ry`.
- The **standardized slope of `ry` on `rx`** is the effect of cosine on the outcome **after
  removing strength**. (This "residualize the predictor *and* the outcome on the controls" is
  Dekker's *double semi-partialling*; it behaves well even when predictor and control are
  themselves correlated.)
- For significance: permute the **nodes of the predictor**, re-residualize, recompute the
  slope → exact node-permutation null, same as Mantel.

### 2.6 What "standardized partial slope" (β) means

Because both sides are standardized, the slope **equals the partial correlation** between
cosine and the outcome, controlling for strength. So **β ∈ [−1, +1]**: its **sign** is the
direction of the effect (positive = aligned-reinforce / opposed-suppress), and its **magnitude**
is an effect size. β = +0.54 is a strong partial correlation. The **bivariate r** we also
report is the same quantity *without* the strength control.

### 2.7 Why we run it *per scheme*, never pooled

Two reasons. (i) Pooling re-introduces Problem B (the same cosine counted twice). (ii) The two
schemes deliver the dose differently (§3.3), so a pooled coefficient mixes two mechanisms.
Each scheme gives a **clean, independent** test on its own 28- (or 22-) edge matrix. We report
them **side by side and never average them into one number** — they are answering slightly
different questions (see §4.6).

### 2.8 Handling gated (missing) dyads under permutation

The coherence gate (§3.6) deletes whole dyads: `per_axis` loses 6 of 28; `normalized_sum`
loses none. A deleted dyad is `NaN` in *every* matrix for that scheme. Under a node
relabelling those `NaN` cells move around, so we use **available-case** node permutation: for
each relabelling the statistic is computed on the dyads that are observed in **both** the
(permuted) predictor and the outcome, refitting the control each time. This is the standard QAP
treatment of structurally missing dyads, and it **reduces exactly** to the simple fixed-mask
test when nothing is gated (`normalized_sum`) — an assertion in the code verifies this. The
observed statistic always uses the full set of valid dyads.

---

## 3. The data, explained (not just paths)

### 3.1 Behaviours and steering vectors

Eight behaviours: **apathetic, confidence, evil, formality, hallucinating, humorous, impolite,
sycophantic.** Each has a unit steering vector `v̂` extracted at layer 17. Adding `α · v̂` to the
residual stream steers the model toward that behaviour; `α` is the dose.

### 3.2 The four injection settings and the deltas

For each pair `(i, j)` we run four conditions and have a judge (`gpt-4.1-mini`) score, 0–100,
how strongly each behaviour appears in the output (and how *coherent* the output is):

| setting | what is injected | notation |
|---|---|---|
| (0,0) | nothing (baseline) | `E_i^(0,0)` |
| (1,0) | only behaviour i | `E_i^(1,0)` |
| (0,1) | only behaviour j | `E_j^(0,1)` |
| (1,1) | both (the composition) | `E_i^(1,1)`, `E_j^(1,1)` |

From these we form two **deltas** per behaviour (how much steering raised its score above
baseline):

- **Individual effect:** `Δ_i^ind  = E_i^(1,0) − E_i^(0,0)` — behaviour i when steered **alone**.
- **Joint effect:** `Δ_i^joint = E_i^(1,1) − E_i^(0,0)` — behaviour i when steered **with j**.

The key difference, `Δ_i^joint − Δ_i^ind = E_i^(1,1) − E_i^(1,0)`, is **how i's expression
changes when you add j's steering on top of i's**: positive = j *boosts* i (reinforcement),
negative = j *dampens* i (suppression).

### 3.3 The two composition schemes (and why two)

Both schemes are **dose-controlled** — they fix the injection magnitude so that cosine is not
mechanically confounded with raw dose (the naive `v1 / normFalse` scheme, where it is, is
excluded). They differ in *how* the two unit vectors are combined into the joint injection
`δ` at setting (1,1):

| scheme (data name) | paper name | joint `δ` | per-axis dose to each behaviour | role |
|---|---|---|---|---|
| `normTrue` (v2) | **normalized_sum** | `α·(v̂_i+v̂_j)/‖v̂_i+v̂_j‖` | `α·√((1+cos)/2)` — **grows with cos** | committed **primary** dataset |
| `per_axis` (v3) | **per_axis** | `(α/(1+cos))·(v̂_i+v̂_j)` | `α` — **constant** | **mechanical-null** dataset |

This difference matters for interpretation:

- **`per_axis` is the mechanical null.** By construction its first-order *projection* shift
  between joint and single is **exactly zero** — each behaviour gets the same push it would get
  if steered alone. So if the judge *still* sees a cosine-dependent composition effect under
  `per_axis`, that is evidence of a **genuine (non-mechanical) interaction**. This is the
  conservative, mechanism-clean test.
- **`normalized_sum` mechanically over-doses aligned pairs.** Its per-axis dose
  `α·√((1+cos)/2)` rises with cosine, so higher-cos pairs get pushed harder in the joint
  condition. Part of any `Smat`-vs-cos relationship under `normalized_sum` is therefore
  *mechanical*, not interaction. We **cannot** cleanly separate the two within this scheme
  (cosine and the mechanical push are deterministically linked), so we **do not** add the
  mechanical push as a control — it would absorb the very signal we are testing. We control
  only `STR`, which is safe in both schemes.

So `normalized_sum` gives **statistical power** (all 28 pairs, no gating) but **mechanical
entanglement**; `per_axis` gives **mechanism-clean** evidence but **less power** and a gated
opposed side. They are complementary, not redundant.

### 3.4 The geometric predictor

`COS[i,j]` = **signed** cosine between `v̂_i` and `v̂_j`. Positive = aligned, negative = opposed.
It is purely geometric and identical across schemes.

### 3.5 The three matrices the test uses

With the deltas above:

- **Direction (main outcome):**
  `Smat[i,j] = ½·[(Δ_i^joint − Δ_i^ind) + (Δ_j^joint − Δ_j^ind)]`
  — averaged over the two behaviours, **> 0 = reinforcement, < 0 = suppression.**
- **Magnitude (second outcome):**
  `Mmat[i,j] = ½·(|Δ_i^joint| + |Δ_j^joint|)`
  — how strongly the behaviours express jointly, in absolute terms.
- **Strength (control):**
  `STR[i,j] = ½·(|Δ_i^ind| + |Δ_j^ind|)`
  — how strongly each behaviour steers **alone**; captures "these are just strong behaviours,"
  independent of geometry.

### 3.6 The coherence gate, and what `per_axis` loses

The judge also rates **coherence** (0–100). If steering breaks the model into gibberish, the
behaviour scores are meaningless, so we drop any dyad with **coherence < 30** (this gate is
already baked into the data we use). Result:

- **`normalized_sum`: 0 dyads gated** — all 28 pairs survive (dose held at `α`, coherence safe).
- **`per_axis`: 6 dyads gated** (22 survive). Its dose `‖δ‖ = α·√(2/(1+cos))` **blows up as
  cos → −1**, so opposed pairs decohere. The 6 lost dyads, with their cosines, are:

  | gated dyad | cos | coherence |
  |---|---|---|
  | formality–humorous | −0.522 | 6.3 |
  | apathetic–hallucinating | −0.164 | 29.9 |
  | confidence–humorous | −0.080 | 26.4 |
  | hallucinating–impolite | −0.051 | 23.3 |
  | hallucinating–humorous | +0.131 | 23.2 |
  | evil–humorous | +0.368 | 28.3 |

  **4 of the 6 have cos ≤ 0** — `per_axis` loses most of its *opposed* side. So its directional
  estimate leans on the *aligned* (reinforce) side; the **suppression** evidence rests on
  `normalized_sum`, which keeps every opposed pair. (The losses concentrate on `humorous` and
  `hallucinating` — visible as the column/row of ×'s in the matrix figure.)

### 3.7 Sanity checks that the data is what we think

- **Cosine is shared across schemes:** max |Δcos| = 0.0 (confirms Problem B was real).
- **Single-injection settings (1,0)/(0,1) are scheme-independent:** the solo deltas match across
  schemes at r ≈ 0.99 (mean absolute difference ~2–3 judge points — pure sampling noise, since
  they are nominally the same condition measured in two runs). This justifies treating `Δ^ind`
  (and hence `STR`) as a scheme-independent control.
- Code asserts: all matrices symmetric; ≤ 28 edges; identity permutation reproduces the observed
  β; 0 ≤ p ≤ 1; fast path == literal Dekker formula.

---

## 4. Results, figure by figure

### 4.0 The headline numbers

Standardized partial slope `β` (= partial correlation, controlling `STR`), with **exact**
two-sided p over all 40,320 permutations:

| outcome | predictor | control | normalized_sum (28 edges) | per_axis (22 edges) |
|---|---|---|---|---|
| **direction** `Smat` | `COS` | `STR` | **β = +0.54, p = 0.009** | **β = +0.52, p = 0.051** |
| direction `Smat` | `|COS|` | `STR` | β = +0.02, p = 0.92 | β = +0.23, p = 0.41 |
| magnitude `Mmat` | `COS` | `STR` | β = +0.51, p = 0.015 | β = +0.40, p = 0.14 |
| direction `Smat` | `COS` | — (bivariate r) | r = +0.48, p = 0.018 | r = +0.47, p = 0.064 |
| magnitude `Mmat` | `COS` | — (bivariate r) | r = +0.65, p = 0.002 | r = +0.64, p = 0.016 |

### 4.1 Figure — the data the test runs on

![fig_mrqap_matrices](../analysis/figures/mrqap/fig_mrqap_matrices.png)

The six 8×8 matrices, one row per scheme. `COS` (red = aligned, blue = opposed), `Smat`
(red = reinforce, blue = suppress), `Mmat` (magnitude). Grey diagonal is ignored; **× = a dyad
removed by the coherence gate.** Two things to see: (i) `normalized_sum` (top) is complete;
`per_axis` (bottom) has 6 ×'s clustered on `humorous`/`hallucinating` — the gated, mostly-opposed
side. (ii) Compare the `COS` and `Smat` columns by eye: the blue (suppression) cells in `Smat` —
e.g. `formality–humorous`, `confidence–humorous` — sit where `COS` is blue (opposed). That visual
correspondence is exactly what the Mantel/MRQAP test quantifies. The **upper triangles** of these
matrices are the vectors fed to the permutation test.

### 4.2 Figure — the headline, and the sign check

![fig_mrqap_direction_scatter](../analysis/figures/mrqap/fig_mrqap_direction_scatter.png)

Each point is one behaviour pair; y = `Smat` (above the dashed line = reinforcement, below =
suppression). **Left column (vs signed cosine):** a clear positive slope in both schemes —
aligned pairs reinforce, opposed pairs suppress (β|STR = +0.54, p = 0.009 for normalized_sum;
+0.52, p = 0.051 for per_axis). **Right column (vs |cosine|):** essentially flat
(normalized_sum r = +0.02, p = 0.92). The contrast is the important part: it is the **sign** of
the cosine (aligned vs opposed) that drives the outcome, **not** the magnitude of similarity. A
"behaviours that are more *similar in either direction* compose more" story would show up in the
right column; it doesn't. Note `per_axis`'s left panel leans on the right (aligned) half of the
x-axis — that is the gated opposed side missing.

### 4.3 Figure — what the exact test actually computes

![fig_mrqap_perm_null](../analysis/figures/mrqap/fig_mrqap_perm_null.png)

The histogram **is** the null distribution: the standardized partial slope under **all 40,320**
relabellings of the 8 behaviours. Black line = the observed slope; coloured bars = the two-sided
tail (|β| ≥ |β_obs|); that tail's share of the 40,320 permutations **is** the exact p. For
`normalized_sum` the observed slope sits far out in the right tail (p = 0.009). For `per_axis` it
sits right at the 5 % edge (p = 0.051). This picture is the entire inferential claim — and it is
why the test is honest: the null is generated by relabelling behaviours, so it bakes in the
dyadic dependence that the old edge-shuffle ignored.

### 4.4 Figure — the magnitude outcome (and a scheme split that matters)

![fig_mrqap_magnitude_scatter](../analysis/figures/mrqap/fig_mrqap_magnitude_scatter.png)

`Mmat` (joint magnitude) vs signed cosine. Both schemes show a strong **bivariate** positive
slope (r ≈ 0.65, p = 0.002 for normalized_sum; r = 0.64, p = 0.016 for per_axis). **But the
controlled story splits by scheme**, and this split is informative:

- **normalized_sum:** even after controlling `STR`, magnitude still rises with cosine
  (β = +0.51, **p = 0.015**). This is expected — recall normalized_sum **mechanically over-doses
  aligned pairs** (§3.3), so aligned pairs simply receive a bigger joint push and express more.
- **per_axis (mechanical null):** once `STR` is controlled, the cosine–magnitude link is **not
  significant** (β = +0.40, **p = 0.14**). With the mechanical over-dosing removed by design, and
  individual strength partialled out, magnitude is effectively **null** with respect to geometry.

So the right reading is not "magnitude tracks geometry" full stop, but: *the apparent
magnitude–cosine link is largely mechanical/strength-driven; under the mechanism-clean scheme it
washes out.* Magnitude is a near-null **where it should be** (per_axis), and the residual
normalized_sum effect is the expected mechanical dose. Report this honestly rather than as a
second geometric finding.

### 4.5 Figure — how much the effect leans on single behaviours

![fig_mrqap_loo_leverage](../analysis/figures/mrqap/fig_mrqap_loo_leverage.png)

Leave-one-behaviour-out: each row drops one behaviour (7 nodes, 5,040 permutations) and refits
the headline `Smat ~ COS | STR`; the diamond is the full 8-behaviour estimate, the dashed line
its β; **filled = p < 0.05, open = p ≥ 0.05.** For `normalized_sum` the effect is robust — it
stays below 0.05 in **6 of 8** folds (a 7th, dropping `humorous`, sits exactly at p = 0.050) and
collapses only when **`formality`** is removed (β → 0.25, p = 0.32), telling us `formality`
carries much of the suppression signal (it has the most opposed pairs). For `per_axis` the effect
is **fragile** — most folds land above 0.05. This is the
expected verdict for a borderline (p ≈ 0.05) result on 22 edges, and it is worth stating: the
`per_axis` evidence is suggestive, not robust to dropping individual behaviours.

### 4.6 What survived, and what to claim in the paper

- **Direction is predicted by signed cosine.** Aligned behaviours reinforce, opposed behaviours
  suppress. Significant under the primary scheme (p = 0.009), borderline under the mechanism-clean
  scheme (p = 0.051). It is the **sign**, not the magnitude, of cosine that matters (`|cos|` is
  null).
- **The two schemes are complementary, not a single number.** `normalized_sum` has power but a
  built-in cosine-dependent mechanical dose; `per_axis` is mechanical-null but loses most opposed
  pairs to the coherence gate. The **suppression** half of the claim rests mainly on
  `normalized_sum`; the **mechanism-clean** confirmation rests on `per_axis`. Neither alone is
  decisive; together they make a consistent but **modest** case.
- **Magnitude is not a competing geometric effect.** Its cosine link is mechanical/strength-driven
  and vanishes under `per_axis` after controlling strength.
- **Honesty about effective sample size.** With 8 behaviours the test has at most 28 (or 22)
  independent dyads and an 8! permutation space — *not* 76 rows. The corrected p-values are larger
  than our first report precisely because the first report borrowed confidence from dyadic
  dependence. The deliverable is the clean, dependence-respecting p, whatever its size.

---

## 5. Are the p-values correct? Validation and calibration

A permutation test is only as trustworthy as (a) its arithmetic and (b) its null model. We can
**prove** (a) and **measure** (b), and we are explicit about what neither settles.
`analysis/rq1_mrqap_validate.py` runs every check below against the *same* `mrqap_exact`
function that produced the reported numbers.

### 5.1 The computation is correct (four checks)

1. **The permutation group is complete and unique** — exactly 8! = 40,320 distinct relabellings
   are enumerated (identity first). "Exact" means the *whole* group; nothing is missing or
   double-counted.
2. **The observed β matches an independent recomputation** — the standardized partial slope from
   `mrqap_exact` equals a from-scratch `scipy.stats.pearsonr` on `numpy.lstsq` residuals (a code
   path that shares nothing with the test's internals): normalized_sum +0.536280 vs +0.536280
   (difference 0); per_axis to 1e-16.
3. **It is deterministic** — identical inputs give byte-identical p on re-run (normalized_sum
   0.008978, per_axis 0.051190). The exact p is a fixed rational `count / 40,320`, no hidden
   randomness.
4. **Known-answer test** — predictor = outcome (perfect association) returns β = 1.000 and
   p = 0.000025 = exactly 1/40,320 (the floor: only the identity permutation matches);
   predictor = −outcome gives the same p, confirming two-sidedness.

Together these rule out the usual failure modes: a wrong statistic, an off-by-one in the count,
omitting the observed value, broken two-sidedness, or non-determinism.

### 5.2 The test is calibrated (type-I error)

A valid permutation test must return a **Uniform(0,1)** p-value when the predictor is unrelated
to the outcome. We fed hundreds of random null predictors (independent of the outcome, with and
without realistic node structure) through `mrqap_exact` and inspected the p-values:

| scheme | null predictor | K | mean p | frac ≤ 0.05 | KS p vs uniform | reading |
|---|---|---:|---:|---:|---:|---|
| normalized_sum | iid | 250 | 0.497 | 0.072 | 0.810 | **calibrated** |
| normalized_sum | node-structured | 150 | 0.503 | 0.067 | 0.919 | **calibrated** |
| per_axis | iid | 130 | 0.597 | 0.015 | 0.001 | **conservative** |

- **normalized_sum is exactly calibrated** (mean p ≈ 0.50, KS consistent with uniform). When
  there is no real association the test does not produce small p-values — so **p = 0.009 is
  trustworthy at face value.**
- **per_axis is conservative** — its null p-values skew *high* (mean 0.60, ~4 SE above 0.5; only
  1.5 % fall below 0.05; KS rejects uniformity *in the safe direction*). The cause is the gated
  dyads: the observed statistic uses all 22 valid edges, but many permutations use fewer (a `NaN`
  is relabelled into the masked-in set), which widens the permuted-slope distribution and makes
  the observed look *less* extreme. The consequence is reassuring: **per_axis p = 0.051 is an
  upper bound on the false-positive rate** — the available-case handling errs toward
  *harder*-to-detect significance, never toward inventing it.

So **neither reported p is inflated**: normalized_sum is dead-on, per_axis is conservative. That
is the opposite of the "borrowed confidence" that sank the original pooled test (§1).

### 5.3 What validation cannot settle

- **The null is a modelling choice.** Node-exchangeability is the correct null for dyadic data
  (the whole argument of §1), but it is a choice, not a theorem.
- **p is conditional on the measured matrices.** The judge scores carry sampling noise (5
  generations per question); the test treats `Smat`, `COS`, … as fixed and does not propagate
  that measurement error. Standard, but it means each p is "given these measured cells."
- **per_axis leans on the method more.** Its gate correlates with the predictor, so the
  available-case handling is more assumption-laden than normalized_sum's clean full-mask test —
  which is exactly why per_axis is the borderline scheme and why we report the two schemes
  separately rather than pooled.

---

## 6. Suggested wording for the paper

> *Methods.* Because behaviour pairs are dyadic — eight behaviours generate 28 non-independent
> pairs, and each behaviour's steering vector recurs in seven of them — we test the
> geometry→composition association with an exact node-permutation test (Mantel / MRQAP with
> Dekker double semi-partialling) rather than a pooled regression over pair-level rows. For each
> dose-controlled scheme we form symmetric 8×8 behaviour matrices of the signed cosine
> (predictor), the directional composition outcome, the joint magnitude, and the individual
> steering strength (control). We enumerate all 8! = 40,320 relabellings of the behaviours to
> obtain an exact two-sided p for the standardized partial slope of the outcome on cosine,
> controlling for strength. We verified the procedure is calibrated under random null predictors
> (uniform p) for the normalized-sum scheme and conservative for per-axis, where coherence-gated
> dyads make the available-case test under-reject.

> *Results.* The signed cosine predicts the **direction** of composition — aligned behaviours
> reinforce, opposed behaviours suppress — with a standardized partial correlation of +0.54
> (exact p = 0.009) under the primary normalized-sum scheme and +0.52 (p = 0.051) under the
> mechanical-null per-axis scheme. The effect is carried by the **sign** of the cosine (|cos| is
> null, p = 0.92), consistent with a directional rather than a magnitude-of-similarity mechanism.
> Joint magnitude correlates with cosine only through the mechanical dose and individual strength:
> after controlling strength it remains significant under normalized-sum (the scheme that
> over-doses aligned pairs, p = 0.015) but is non-significant under the mechanical-null scheme
> (p = 0.14). A leave-one-behaviour-out analysis shows the directional effect is robust under
> normalized-sum (below 0.05 in 6/8 folds, failing only when formality — which carries the opposed
> pairs — is dropped) but fragile under per-axis.

---

## 7. Reproduce

```bash
./venv/bin/python analysis/rq1_mrqap_significance.py   # tables + asserts → results .md/.csv
./venv/bin/python analysis/rq1_mrqap_plots.py          # the five figures → analysis/figures/mrqap/
./venv/bin/python analysis/rq1_mrqap_validate.py       # computation + calibration checks (§5)
```

Data in = the coh ≥ 30 frame used by `analysis/notebooks/rq1_v2v3_decomposition.ipynb`
(`analysis/rq1_consolidated/consolidated_coh30.csv`); gate removals confirmed against
`consolidated_raw.csv`. The test itself lives in `analysis/rq1_mrqap_significance.py`
(`mrqap_exact`, `partial_slope`); the plotting reuses that exact machinery so figure numbers
match the tables.
