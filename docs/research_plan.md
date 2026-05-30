# Research Proposal: Predicting Behavior Composition from Steering-Vector Geometry

# Research Questions

This project investigates whether the geometric relationship between two behavioral steering vectors in the residual-stream activation space of a language model predicts how those behaviors compose when the vectors are applied jointly, and — given such a predictive rule — what mechanism in the network produces the observed interference. We study these questions on a fixed set of **eight** behaviors (apathetic, evil, hallucinating, humorous, impolite, sycophantic, confidence, formality) spanning safety, style, and persona axes, using mean-difference steering vectors extracted at a single validation-selected layer of Llama-3.1-8B from contrastive pairs drawn from the Model-Written Evaluations datasets. The initial twelve-behavior set was narrowed during validation: three behaviors (corrigibility, myopia, verbosity) failed the Phase-7 dual-signal validation (judge × logprob); refusal and agreeableness were dropped on grounds of evaluation reliability; and power-seeking was dropped at Phase 12.5 after the calibration audit (E15.13) showed an unstable trait × judge × prompt interaction that produced unusable Δ-ratios in every normalisation mode tested.

##### RQ1 — Structure

Does the pairwise geometric relationship between two steering vectors predict how the corresponding behaviors compose under joint steering?

A known but unresolved difficulty in multi-behavior steering is that naive linear combination of individual steering vectors often produces outcomes that are not simply the sum of the individual effects: one behavior may dominate, both may be suppressed, or an unintended third behavior may emerge. Whether this phenomenon is systematically predictable from the geometry of the two vectors — and therefore whether a practitioner could, in principle, know in advance which behaviors compose cleanly and which do not — is an open empirical question. RQ1 asks whether the pairwise angle between two steering vectors, in the simplest case their cosine similarity, predicts the functional outcome of applying both vectors jointly in the instruction-tuned model. We decompose this into four sub-questions: how the selected behavior pairs distribute across the four possible composition regimes (additive, dominant, mutually suppressive, emergent); whether pairwise cosine similarity alone predicts whether a pair composes additively, and with what reliability; whether a continuous measure of composition quality correlates monotonically with pairwise cosine; and whether a richer geometric predictor that accounts for shared loadings on the leading principal components of the full behavior set outperforms simple pairwise cosine.

##### RQ2 — Mechanism

When two steering vectors compose non-additively in the instruction-tuned model, where in the network does the interference occur, and does the per-layer interference signature differ systematically between pairs that compose additively and pairs that do not?

RQ1 produces a predictive rule relating geometry to compositional outcome, but it does not explain why the rule works. RQ2 asks whether the predictive rule can be grounded in a mechanism: specifically, by tracking how the residual-stream activations at layers downstream of  $L^*$  project onto the two steering vectors, we can observe how the  $L^*$  steering signal propagates or decays through the rest of the network under joint intervention. We decompose this into four sub-questions: (2a) how the projections  $\pi_i(L) = \langle h^{(L)}, v_i^{(L^*)} \rangle / \|v_i^{(L^*)}\|$  and  $\pi_j(L)$  evolve across downstream layers  $L > L^*$  under joint steering; (2b) whether additive pairs and non-additive pairs exhibit qualitatively different trajectory shapes, with non-additive pairs showing systematic decay or distortion of one or both projections relative to their individually-steered trajectories; (2c) whether the layer at which trajectories diverge from their individual-steering baselines is concentrated in a narrow range or distributed across the network; and (2d) whether a trajectory-based feature predicts compositional regime better than, or complementarily to, raw pairwise cosine. As a closing bridge sub-question, we also ask whether the geometric and mechanistic structure identified in the instruction-tuned model is specific to instruction tuning or inherited from the pretrained base model — addressed in Part C, where time permits.

##### Procedure

##### 1. Setup and behavior selection

We use meta-llama/Llama-3.1-8B (base) and meta-llama/Llama-3.1-8B-Instruct (instruction-tuned). The eight target behaviors (apathetic, evil, hallucinating, humorous, impolite, sycophantic, confidence, formality) are drawn from Anthropic's Model-Written Evaluations persona datasets, supplemented by the standard sycophancy and TruthfulQA-derived hallucination datasets. For each behavior, we use 400 contrastive prompt pairs, split 60/20/20 into training (steering vector extraction), validation (layer selection and coefficient tuning), and test (all reported results). Each contrastive pair consists of an identical prompt followed by two candidate completions, one exhibiting the target behavior and one not exhibiting it; we use the MWE multiple-choice format to ensure minimal confounds between the positive and negative members of each pair.

##### 2. Activation extraction and steering vector construction

For each (model, behavior, training pair), we run a forward pass and extract the residual-stream activation at the final prompt token position, across all transformer layers. We use hook-based extraction (either via transformer\_lens or raw PyTorch forward hooks on the residual stream outputs). For each behavior b and each layer L, let  $A_{+,b}^{(L)} = \{a_{+,1}^{(L)}, \ldots, a_{+,N}^{(L)}\}$  denote the activations from the positive (behavior-exhibiting) completions and  $A_{-,b}^{(L)}$  the corresponding negatives, where N=240 is the training set size. The mean-difference (CAA) steering vector is

$$v_b^{(L)} = \frac{1}{N} \sum_{i=1}^{N} a_{+,i}^{(L)} - \frac{1}{N} \sum_{i=1}^{N} a_{-,i}^{(L)} \in \mathbb{R}^d,$$
 (1)

where d=4096 is the model's hidden dimension. We apply this procedure to the instruct model first, then repeat on the base model at the same layer  $L^{\star}$  selected below. Additionally, once  $L^{\star}$  is selected, we extract per-layer CAA vectors  $v_b^{(L)}$  at every downstream layer  $L>L^{\star}$  on the instruct model, for use in the RQ2 mechanism analysis (Section 6); this is a re-run of the same extraction procedure at additional layer indices and requires no new data.

##### 3. Layer selection

On the validation split of the instruct model, for each behavior and each layer, we apply the steering vector at coefficient +1 (i.e., we add  $v_b^{(L)}$  to the residual stream at layer L during forward passes on held-out prompts) and measure behavior expression on generated completions. We select a single layer  $L^*$  that maximizes the *mean* expression across behaviors, not the per-behavior optimum, to avoid overfitting layer choice to individual behaviors. The selected operating layer is **$L^* = 17$** (1-indexed; corresponds to a forward hook on `model.model.layers[16]`), chosen on the Phase 9 per-trait sweep and confirmed faithful to the Anthropic-paper protocol in Phase 13's §B.4 replication (where the paper's argmax $L=16$ lies within Llama-3.1-8B's coherence-aware plateau at $L \in [16, 18]$; our $L^*=17$ is a within-plateau adaptation). Once selected, $L^*$ is frozen for all subsequent experiments.

##### 4. Behavior expression scoring

We use a single strong LLM-as-judge (`gpt-4.1-mini`) with a per-trait standardized scoring prompt that, given a completion and a behavior description, returns an integer in [0, 100] indicating how strongly the behavior is expressed. Validated via the Phase-7 dual-signal pipeline: each candidate behavior must pass both the judge (Δ_trait > 50 between steered and baseline on the validation set) and an MWE logprob shift (|mean_shift| > 0.5 nats) to be retained. The Phase 12.5 composition sweep uses **`N_per_question = 5` completions per question** on a per-pair held-out 20-question evaluation set (giving 100 judge scores per intervention condition and per pair). The judge protocol is uniform across baseline, single-vector, and joint conditions; alongside the per-trait score we also score *coherence* on a 0–100 scale (separate judge prompt) so that joint conditions where the model decoheres can be flagged. The composition score for a pair is the mean of `trait_a` and `trait_b` judge scores.

##### 5. Part A — Composition experiments on the instruct model (RQ1)

> **Calibration note (Phase 12.5).** The operating point described in this section — composition formula, α value, coefficient settings, regime classifier thresholds — is the result of the calibration history documented in [paper/experiments_log.md](experiments_log.md) Phases 7 → 15. The original Phase-12 protocol used the naïve sum  $h \leftarrow h + \alpha_i v_i + \alpha_j v_j$  at  $\alpha=4$, and the audit in Phase 14 identified a magnitude inflation that drove joint coherence into incoherent territory on high-cosine pairs. The protocol below reflects Phase 12.5's recalibration: a magnitude-bounded normalisation at  $\alpha = 4.5$. The pre-recalibration Phase-12 dataset is preserved untouched as a methodological precursor.

Gram matrix. Let Vinst ∈ R<sup>8×d</sup> be the matrix whose rows are the eight instruct-model steering vectors at layer L*, row-wise unit-normalized. We compute the full 8 × 8 Gram matrix of pairwise cosine similarities,

$$G_{ij} = \frac{v_i^{\top} v_j}{\|v_i\| \|v_j\|}, \tag{2}$$

and visualize it as a hierarchically clustered heatmap.

Pair selection. The full set of  $\binom{8}{2} = 28$  unordered pairs is tractable to evaluate exhaustively at our compute budget, so we study every pair rather than stratified-sample a subset. The 28 pairs span the empirical cosine range  $|G_{ij}| \in [0.00, 0.69]$  (no pair in the 8-trait set exceeds  $|G| = 0.7$), with the cosine distribution naturally covering near-orthogonal, moderate, and high-overlap regimes.

Joint steering. We extract unit-normalised vectors  $\hat{v}_b = v_b^{(L^*)} / \|v_b^{(L^*)}\|$  and apply a magnitude-bounded composition: for each selected pair  $(b_i, b_j)$  and each coefficient setting

$$(w_i, w_j) \in \{(0,0), (1,0), (0,1), (1,1)\},$$

we build the steering perturbation

$$\delta_{i,j}(w_i, w_j) = \alpha \cdot \frac{w_i\,\hat{v}_i + w_j\,\hat{v}_j}{\|w_i\,\hat{v}_i + w_j\,\hat{v}_j\|}, \qquad \alpha = 4.5, \tag{3}$$

and apply it via residual-stream addition at  $L^*$  during the response-token positions:

$$h^{(L^{\star})} \leftarrow h^{(L^{\star})} + \delta_{i,j}.$$

Equation (3) is normalize=True composition: total perturbation magnitude  $\|\delta\| = \alpha$  is held constant across pairs regardless of  $G_{ij}$  (singles  $(1,0)/(0,1)$  collapse to  $\alpha\,\hat{v}_{\text{active}}$ , so the single conditions are also magnitude- $\alpha$ ). This replaces the Phase-12 naïve sum  $h \leftarrow h + \alpha(\hat{v}_i + \hat{v}_j)$ , under which  $\|\delta\|$  varied over [4, 7.4] across the cosine range and drove the over-steering coherence collapse identified in Phase 14. The value $\alpha = 4.5$ was selected from the Phase-15 dose-response pilot as the utility-optimal operating point (utility = Δ-composition × coherence / 100; see E15.11–E15.13). We generate $N_{\text{per-q}} = 5$ completions per question on the per-pair 20-question evaluation set at temperature 1.0 (20 questions × 5 = 100 completions per intervention condition per pair). The sign-asymmetric antipodal probes  $(-1, 1)$  and  $(1, -1)$  from the original plan are deferred to a robustness appendix; the four core settings above span the additive / suppressive / dominant / emergent regimes addressed by RQ1 and feed the regime classifier below. During each joint-steering forward pass, we additionally cache the residual-stream activations at all layers L ≥ L ⋆ for downstream use by the RQ2 mechanism analysis (Section 6).

Regime classification. Let  $E_i(w_i, w_j)$  denote the mean per-trait judge score for behavior  $b_i$ under coefficient setting  $(w_i, w_j)$ , averaged over the 100 completions. Define the per-axis ratios

$$r_i = \frac{E_i(1,1) - E_i(0,0)}{E_i(1,0) - E_i(0,0)}, \qquad r_j = \frac{E_j(1,1) - E_j(0,0)}{E_j(0,1) - E_j(0,0)},$$

where the numerator is the joint-vs-baseline change on trait  $i$  and the denominator is the single-vector change. Ratios are clamped to 0 when the single-vector denominator is essentially zero ($|E_i(1,0) - E_i(0,0)| < 10^{-3}$). For each pair, we classify the (1, 1) outcome via the following decision rules (evaluated in order):

- **Additive** if  $0.7 \le r_i \le 1.3$  AND  $0.7 \le r_j \le 1.3$  (both axes preserve their single-vector effect, within 30%);
- **Suppressive** else if  $r_i < 0.5$  AND  $r_j < 0.5$  (both axes lose more than half of their single effect);
- **Emergent** else if  $r_i > 1.3$  AND  $r_j > 1.3$  (both axes amplify beyond their single effect);
- **Dominant** else if one ratio  $\ge 0.7$  AND the other  $\le 0.3$  (one axis preserved, the other crushed);
- **Mixed** otherwise (one axis strong, one in the 0.3–0.7 in-between zone).

The Phase 14 audit raised a concern that the "emergent" label could be a coherence artefact rather than a real composition phenomenon: on low-coherence joint completions the denominator  $E_i(1,0) - E_i(0,0)$  can collapse for spurious reasons, inflating  $r_i$ . The Phase 12.5 recalibration directly tests this: the emergent count went from 6/36 (Phase 12) to 0/28 (Phase 12.5), supporting the artefact interpretation. The thresholds above were pre-registered in [scripts/compositions/composition_scoring.py](../scripts/compositions/composition_scoring.py) before the Phase 12 sweep and have not been re-tuned since; a planned post-Phase-12.5 follow-up (E15.15 #4) is to re-examine the mixed-bucket fraction (50% in Phase 12.5) given the cleaner ratios at the new operating point.

Cosine as predictor. We fit a logistic regression predicting the binary outcome "additive vs. non-additive" from  $|G_{ij}|$ , report the AUC with bootstrap confidence intervals over 1000 resamples, and compare against a permutation-based random baseline (shuffling pair labels 10,000 times). We separately fit a Spearman rank correlation between  $|G_{ij}|$  and the continuous composition quality score

$$Q(i,j) = \frac{1}{2} \, (r_i + r_j), \tag{4}$$

where  $r_i, r_j$  are the per-axis ratios defined in the regime classifier above. The denominator-clamping rule ( $|E_i(1,0) - E_i(0,0)| < 10^{-3} \Rightarrow r_i = 0$ ) avoids division instabilities directly. Both the binary AUC and the continuous Spearman are computed on the Phase-12.5 dataset (28 pairs); the pre-recalibration Phase-12 results are reported as a methodological appendix to show what the AUC looked like under the over-steered operating point.

Higher-order predictor. Let  $U^{(k)} \in \mathbb{R}^{d \times k}$  be the top-k right singular vectors of  $V_{\text{inst}}$, and let  $\tilde{v}_i = U^{(k)\top} v_i$  be the projection of  $v_i$  onto the top-k principal components. We define a subspace-overlap predictor as the cosine similarity between the elementwise absolute projections  $|\tilde{v}_i|$  and  $|\tilde{v}_j|$, which measures whether two behaviors load heavily on the same principal directions regardless of sign. We compare the predictive power of this measure against raw pairwise cosine via nested logistic regression and a likelihood-ratio test. With  $V_{\text{inst}}$  now  $8 \times d$, we use  $k = 3$  as our primary setting (covers most variance for this trait set), with  $k = 4$  and  $k = 5$  as robustness checks.

Symmetry of interference. Deferred to the robustness appendix. The sign-asymmetric coefficient settings  $(-1, 1)$  and  $(1, -1)$  were not run in the Phase-12.5 main sweep; if the writeup space allows, a follow-up sweep on a small subset of pairs can be added to test whether interference is symmetric under sign flip.

##### 6. Part B — Mechanism analysis of interference (RQ2)

The pairwise-cosine rule in Part A predicts compositional outcome from geometry at layer L ⋆ . Part B asks what happens downstream of L <sup>⋆</sup> under joint steering, and whether the per-layer pattern of activation projections reveals a mechanism that distinguishes additive from nonadditive composition. This analysis uses the same generations and cached activations produced during the Part A composition sweep and requires no additional forward passes for the primary (projection onto v (L⋆) ) analysis.

Interference signatures. For each selected pair (b<sup>i</sup> , b<sup>j</sup> ), each coefficient setting (α<sup>i</sup> , α<sup>j</sup> ), and each downstream layer L ≥ L ⋆ , we define the projection trajectory

$$\pi_i^{(\alpha_i, \alpha_j)}(L) = \frac{\langle h^{(L)}, v_i^{(L^*)} \rangle}{\|v_i^{(L^*)}\|},$$
 (5)

where  $h^{(L)}$  is the residual-stream activation at the final prompt token position at layer L, averaged across the 20 evaluation prompts (per generation seed; we also retain per-prompt trajectories for variance estimation). Symmetrically for  $\pi_j^{(\alpha_i,\alpha_j)}(L)$ . The pair of trajectories  $(\pi_i,\pi_j)$  under joint steering (1,1), compared against the individual-steering baselines (1,0) and (0,1), constitutes the *interference signature* of the pair.

Trajectory divergence (2a, 2b). For each pair, we compute a scalar trajectory-divergence measure,

$$\Delta_i(i,j) = \frac{1}{|\mathcal{L}|} \sum_{L \in \mathcal{L}} \left| \pi_i^{(1,1)}(L) - \pi_i^{(1,0)}(L) \right|, \tag{6}$$

where  $\mathcal{L} = \{L^*, L^* + 1, \dots, L_{\text{final}}\}$  is the set of downstream layers, and analogously  $\Delta_j(i, j)$ . Small  $\Delta$  indicates the joint-steering trajectory of  $b_i$ 's projection tracks its individual-steering trajectory (no interference); large  $\Delta$  indicates substantial deviation. We compare the distributions of  $(\Delta_i, \Delta_j)$  across Part A's regime classes: additive pairs should show small  $\Delta$  for both behaviors, suppressive pairs large  $\Delta$  for both, dominant pairs large  $\Delta$  for the suppressed behavior only. We report these comparisons as boxplots stratified by regime and test significance via Kruskal–Wallis across the four regimes. The trajectory dataset is shipped as [results/composition_trajectories_l17_v2.parquet](../results/composition_trajectories_l17_v2.parquet) (268,800 rows: 28 pairs × 3 settings × 100 prompts × 16 layers × 2 behaviour axes).

Layer of divergence (2c). For each non-additive pair, we identify the earliest downstream layer  $L_{\text{div}}(i,j)$  at which  $|\pi_i^{(1,1)}(L) - \pi_i^{(1,0)}(L)|$  exceeds a pre-registered threshold  $\tau$ . Threshold  $\tau$  is calibrated via a split-half R2 bootstrap on the individual-steering trajectories: for each (pair, axis) we draw 1,000 split-halves of the per-prompt projections, compute  $\max_L |\bar{\pi}_A(L) - \bar{\pi}_B(L)|$, pool across all (pair, axis) cells, take the 95th-percentile, and multiply by a cushion factor of 1.5 (per the RQ2 roadmap). At the Phase 12.5 operating point  $\tau = 0.8297$  ([results/composition_trajectories_l17_v2_tau.json](../results/composition_trajectories_l17_v2_tau.json)); trajectories that never cross  $\tau$  saturate at  $L_{\text{final}}$  in the histogram. Concentration of  $L_{\text{div}}$  in a narrow range (e.g., the first few layers after  $L^*$ ) suggests a localized mechanism; a broad distribution suggests interference is distributed across the network.

Trajectory as predictor (2d). We fit a logistic regression predicting additive vs. non-additive from a trajectory-based feature — specifically,  $\max(\Delta_i, \Delta_j)$  — and compare against the Part A cosine-only baseline via nested logistic regression and a likelihood-ratio test. If the trajectory feature adds predictive power above cosine, this is evidence that the mechanism of interference carries information not fully captured by direction overlap at  $L^*$  alone. We also report a combined (cosine + trajectory) logistic as the best predictive model.

**Dual-projection robustness.** The primary analysis projects downstream activations onto  $v_i^{(L^\star)}$  — measuring "how strongly the  $L^\star$  steering signal persists through the downstream computation." A complementary view projects onto  $v_i^{(L)}$ , the CAA direction extracted at layer L itself — measuring "how strongly the behavior is represented at each layer in that layer's own geometry." The dual-projection check is implemented against the Phase 12.5 trajectory parquet for all 28 pairs at every downstream layer  $L > L^\star$ , using the per-layer CAA vectors extracted in Section 2. The primary analyses above are reported with the  $v_i^{(L^\star)}$  projection; the  $v_i^{(L)}$  version is reported as a robustness check, with the same divergence and predictor analyses recomputed. Agreement between the two tells us the findings are geometry-robust; disagreement is itself an interesting finding (the  $L^\star$  direction may decay while the behavior persists downstream, or vice versa) and we report it explicitly.

Mechanism pilot (Phase 11). The trajectory pipeline was validated end-to-end in the Phase 11 pilot (3 pairs × 1 sample × 16 layers), confirming that projection-based interference signatures do distinguish individual-steering from joint-steering trajectories visibly, and providing the seed split-half analysis that informed the R2 τ-calibration recipe used in (2c).

##### 7. Part C — Base-model bridge (time permitting)

Part C asks whether the geometric and mechanistic structure identified in the instruct model is specific to instruction tuning or inherited from the pretrained base model. The original motivation for restricting composition experiments to the instruct model — that base models do not follow instructions, so judge-based behavior evaluation is unreliable on their generations — still applies to the behavioral analysis. However, the geometric comparisons (C1) and the mechanism analysis (C2) do not require behavior evaluation on generated completions and are therefore tractable on the base model. Part C is scoped as contingent on Week 4 progress: C1 is committed if the core RQ1 and RQ2 analyses complete on schedule, and C2 is attempted only if C1 is complete and Bri is genuinely ahead of schedule. C3 (behavioral composition on the base model via MWE log-probabilities) is listed as explicit future work in the writeup and is not attempted in this project.

C1 — Geometric comparison of Gram matrices. We compute Gbase analogously to Ginst using the eight base-model steering vectors at layer L*. We report ∥Gbase −Ginst∥<sup>F</sup> /∥Ginst∥<sup>F</sup> as a normalized global distance, and the Spearman correlation between the 28 off-diagonal entries of Gbase and Ginst. We visualize both matrices as clustered heatmaps with identical row and column orderings (the ordering being determined by hierarchical clustering of Ginst) to allow direct visual comparison. We compare the empirical distributions of {|Gbase,ij |}i<j and {|Ginst,ij |}i<j via a two-sample Kolmogorov–Smirnov test, report mean pairwise cosine in each model and a paired t-test on the 28 pairs, and plot overlaid histograms. A significant leftward shift in the instruct distribution indicates that instruction tuning systematically orthogonalizes behaviors. Finally, we label each behavior by category (safety, style, persona) and compute within-category and between-category mean cosines in each model,

$$\bar{G}_{\text{within}} = \max_{\substack{i < j \\ c(i) = c(j)}} G_{ij}, \qquad \bar{G}_{\text{between}} = \max_{\substack{i < j \\ c(i) \neq c(j)}} G_{ij}, \tag{7}$$

and compare G¯within −G¯ between across the two models to ask whether instruction tuning strengthens, disrupts, or leaves intact category structure. Statistical significance is assessed via permutation test on category labels. Using the logistic regression fit in Part A, we also generate predicted composition outcomes for all 28 pairs in both models and identify pairs that are predicted to compose cleanly in one model but not the other. This analysis is predictive, not causal: we do not run composition experiments on the base model.

C2 — Mechanism comparison (attempted if C1 complete and schedule permits). A natural extension of the Part B mechanism analysis: compute interference signatures on the base model for the same 28 pairs, under the same four coefficient settings ((0,0), (1,0), (0,1), (1,1)), using base-model steering vectors v base,(L⋆) i . No generation or judge evaluation is required, only forward passes measuring the projection trajectories πi(L) — the same activation-space quantities defined in Section 6, now on the base model. We ask: do the same pairs show the same divergence layers and similar trajectory shapes across the two models, or does instruction tuning reshape the mechanism? This sharpens the base-vs-instruct comparison from a summary-statistic level (Gram matrices) to a mechanism level (per-layer propagation of steering perturbations). Given the modest compute cost (28 pairs × 4 settings × 20 prompts × one forward pass each, with no generation), C2 is roughly half a day of work and is attempted if the Part B analyses complete on schedule.

C3 — Behavioral composition on the base model (future work). A fully symmetric base-vs-instruct comparison would require a behavior-expression measurement tool that works on base models. The MWE multiple-choice format, which measures behavior by comparing log-probabilities of the two candidate completions, is such a tool and is the same format used to extract our steering vectors. Running the Part A composition sweep on the base model using MWE log-prob scoring, with a complementary log-prob-vs-judge validation on the instruct model to establish that the two measurement modalities agree, would complete the base-vs-instruct story at the behavioral level. We do not attempt this in the current project, both because the instruct-model log-prob validation introduces a parallel evaluation pipeline whose results may differ from the judge-based pipeline (itself a methodological question worth its own analysis) and because the scope is too broad for the four-week timeline. We flag this as the natural next step and specify it fully in the writeup's future-work section.

##### 8. Robustness

We rerun the Part A logistic regression and the Part C Gram-matrix comparison at layers L <sup>⋆</sup> − 3 and L <sup>⋆</sup> + 3 to verify that the findings are not artifacts of the specific layer choice. We rerun the Part A regression with the judge's scores perturbed by ±5 points per completion (a pessimistic model of judge noise) to check that the predictive AUC remains above the random baseline. We re-score a random 20% subsample of Part A completions with a second judge model and recompute the logistic regression; agreement above AUC = 0.7 with the primary result is our acceptance criterion. For Part B, the dual-projection robustness check (Section 6) directly tests whether the mechanism findings survive under an alternative choice of projection direction.

## Timeline and Plan

> **Note (Phase 12.5).** The timeline below is the ex-ante four-week plan as written at project start. Actual execution ran longer and was punctuated by two re-calibration passes (Phase 7 trait narrowing and Phase 14/15 normalisation recalibration); see [paper/experiments_log.md](experiments_log.md) for the phase-by-phase chronology. The pair-count and coefficient-setting figures in the daily breakdown below (40 pairs × 6 settings × 25 completions ≈ 120,000 generations) reflect the original scope; the executed Phase-12.5 scope is 28 pairs × 4 settings × 5 completions × 20 questions ≈ 11,200 generations + ~270,000 trajectory rows.

The project has four roles: Infrastructure Lead (Inf), Analysis Lead (Ana), Evaluation Lead (Eval), and Bridge/Mechanism Lead (Bri). Ana also acts as project captain, owning the overall timeline, figure list, and writeup skeleton. Bri owns RQ2's mechanism analysis and Part C, and operates largely in parallel with the other roles: Inf and Ana finalize L <sup>⋆</sup> and the eight instruct-model steering vectors at L ⋆ ; Bri then re-runs Inf's extraction code at additional model/layer indices to produce the base-model vectors and the per-downstream-layer vectors needed for the dual-projection robustness, and builds out the mechanism analysis pipeline against the cached activations from the Part A sweep.

#### Week 1 — Infrastructure and steering vectors

- Days 1–2 (joint). Environment and GPU setup, repository conventions, joint reading of the five Tier-1 papers, final lock on the twelve behaviors and on the shared data schemas (activation tensor shape, steering-vector record, judge-score record, composition-sweep row, trajectory record).
- Days 3–5 (parallel). Infrastructure Lead builds the activation-extraction pipeline on the instruct model and extracts activations for all 400 pairs per behavior, across all layers. Analysis Lead implements and unit-tests the utilities for pairwise-cosine computation, logistic regression with bootstrap confidence intervals, Kolmogorov–Smirnov testing, and permutation testing on synthetic data. Evaluation Lead builds and validates the LLM-judge against 30 hand-labeled examples per behavior on three behaviors, targeting Spearman > 0.7, and sets up the secondary-judge pipeline for robustness. Bridge/Mechanism Lead reads the Tier-1 papers deeply (especially Chen et al. Persona Vectors and Arditi et al. Refusal) alongside priority Tier-2 reading (Elhage et al., Geva et al.), implements and unit-tests the per-layer projection utility πi(L) and the trajectory divergence metric ∆ on synthetic data, and prepares the basemodel extraction pipeline (same code as Inf's instruct-model pipeline, different checkpoint) so that it is ready to run the moment L ⋆ is selected.
- End of Week 1. CAA vectors computed for the instruct model across all layers; layer L ⋆ selected and frozen; individual-behavior steering sanity check passed. Bri then runs (i) the

base-model extraction at L <sup>⋆</sup> and (ii) the per-downstream-layer extraction on the instruct model for all L > L<sup>⋆</sup> — both are reruns of Inf's extraction code at additional model/layer indices and together take roughly one day of compute.

##### Week 2 — Part A setup: Gram matrix, pair selection, pilot

- Days 1–2. Compute and visualize the instruct Gram matrix Ginst. Select the 40 stratified pairs. In parallel, Bri computes Gbase, produces the side-by-side clustered heatmap with shared ordering, and runs the C1 geometric analyses (KS test, Frobenius, paired t-test, category-cluster permutation test); this completes most of Part C1 by end of Day 2.
- Days 3–4. Run a five-pair pilot of the joint-steering protocol across all six coefficient settings. Inspect judge outputs, calibrate regime-classification thresholds, pre-register them in the repository. In parallel, Bri extracts pilot-pair trajectories from the cached activations, inspects them by eye, and runs the mechanism go/no-go: if pilot trajectories show visible structure across coefficient settings, calibrate the Ldiv threshold and pre-register; if they look like noise, commit to proceeding with the full sweep and reporting the result either way.
- Day 5. Extract activations and compute CAA vectors for the base model at layer L ⋆ . Compute Gbase. This is the entire base-model contribution and takes roughly one day. Bri finalizes the trajectory-analysis notebook so it is ready to consume Week 3's streaming data.
- End of Week 2. Gram matrices for both models computed; composition pilot complete; thresholds pre-registered; mechanism pipeline tested end-to-end on pilot data; Part C1 essentially complete.

##### Week 3 — Part A: Full composition sweep

- Days 1–4. Run the full composition sweep: 40 pairs × 6 coefficient settings × 20 prompts × 25 completions, totaling approximately 120,000 generations. Judge-scoring runs in parallel as generations complete, with the secondary judge re-scoring a 20% subsample. The extraction pipeline caches residual-stream activations at all L ≥ L <sup>⋆</sup> alongside the generations, so trajectory data accumulates automatically; Bri runs streaming trajectory analyses that do not require regime labels (per-pair trajectory visualizations, Ldiv distributions, unsupervised clustering of trajectory shapes) and drafts the Part B and Part C writeups against preliminary data.
- Day 5. Classify regimes and compute continuous composition-quality scores Q(i, j) for all 40 pairs.
- End of Week 3. Full Part A dataset collected and classified. Trajectory dataset complete. Label-free mechanism analyses produced. Part B and Part C drafts underway.

##### Week 4 — Analysis, bridge, robustness, writeup

• Days 1–2. Fit the Part A logistic regression (|Gij | predicting additive vs. non-additive), report AUC with bootstrap confidence intervals and permutation baseline. Fit the Spearman correlation between |Gij | and Q(i, j). Fit the higher-order predictor and compare via likelihoodratio test. Test symmetry of interference across (−1, 1) and (1, −1). Produce the Part A headline figure: pairwise cosine on the x-axis, composition quality on the y-axis, regimecolored points, logistic regression fit with confidence band. In parallel, Bri fits the Part B logistic regression (trajectory feature plus cosine vs. cosine-only), runs the likelihood-ratio test, produces the Part B headline figure (per-layer projection trajectories for additive vs. non-additive pairs with confidence bands), and runs the dual-projection robustness check — recomputing trajectories using v (L) i at every downstream layer and rerunning the divergence and predictor analyses for comparison against the primary v (L⋆) i results.

- Day 3. Run the Part C1 bridge analysis: Gram-matrix comparison (Frobenius, Spearman), Kolmogorov–Smirnov test on cosine distributions, paired t-test on mean cosine, predicted-pair differences across the two models (using the Part A regression on both Gram matrices), category cluster structure and permutation test. Produce side-by-side clustered heatmap of Gbase and Ginst. If schedule permits (which is expected for the easy parts of Part C), Bri also computes base-model interference signatures for the 40 pairs (C2) and produces the side-by-side instruct-vs-base mechanism figure.
- Day 4. Robustness checks at L <sup>⋆</sup> ± 3, under perturbed judge scores, and with the secondary judge.
- Day 5 and final weekend. Writeup divided by role and drafted in parallel, with Infrastructure Lead writing methods, Analysis Lead writing Parts A and C1 results, Evaluation Lead writing the evaluation protocol and judge validation appendix, and Bridge/Mechanism Lead writing Part B results, Part C results, and the related-work section. Final integration pass, figure polishing, abstract, limitations section, future-work section (with C3 specified as the natural extension), submission draft.

##### Fallback

If the Week 2 pilot reveals noise or the Week 3 composition sweep runs behind schedule, we reduce scope in the following priority order, designed so the first cuts do not affect core claims:

- 1. Drop Part C2 (base-model mechanism analysis) entirely, keeping Part C1 (geometric comparison, inherited from original scope) as the only base-model contribution. This is already pre-committed: C2 is attempted only if schedule permits.
- 2. Drop the dual-projection robustness check; revert to v (L⋆) i projection only and note as a limitation.
- 3. Drop sign-asymmetric coefficient settings (−1, 1) and (1, −1) from Part A, keeping only the four core settings (0, 0),(1, 0),(0, 1),(1, 1).
- 4. Reduce from 40 pairs to 30, removing approximately three pairs from each stratum.
- 5. Reduce completions per prompt from 25 to 15.

Each reduction preserves the core findings while freeing approximately one day. If deeper reductions are required, the project reduces to Part A plus a light Part B (30 pairs, four coefficient settings, primary projection only) and Part C1 only — still producing a reportable finding on both RQ1 and RQ2.

##### Reading list

Organized by priority. Tier 1 is mandatory for all four team members before the end of Week 1. Tier 2 is role-dependent and should be split across the team during Weeks 1–2. Tier 3 is consulted during writeup and for situating the related-work section.

# Tier 1 — Mandatory for all team members, Week 1

These define the conceptual and methodological foundation of the project. Budget one day for joint reading and a 90-minute discussion meeting at the end. None of these can be skipped.

- 1. Panickssery et al. 2024 "Steering Llama 2 via Contrastive Activation Addition." The CAA paper. Read carefully: contrastive-pair construction, mean-difference formula, layer selection protocol. The entire steering-vector pipeline is built on this. Pay particular attention to the coefficient sweeps and the validation that a vector actually steers. arXiv: 2312.06681
- 2. Weij, Hofstätter, Jaffe, Brown, Ward 2024 "Extending Activation Steering to Broad Skills and Multiple Behaviours." The single most important paper for this project. Directly motivates RQ1 by documenting that naive linear combination of steering vectors fails, and proposing that applying different vectors at different layers works better. The failure modes described here define what "interference" means in our composition experiments, and the partial solutions establish what we are trying to improve on via the geometric-prediction angle.

arXiv: 2403.05767

- 3. Chen, Arditi, Sleight, Evans, Lindsey 2025 "Persona Vectors: Monitoring and Controlling Character Traits in Language Models" (Anthropic). The closest methodological neighbor. Automated pipeline for extracting steering vectors for non-emotional traits spanning exactly the behavior categories we are using. Read for: the base-vs-instruct observations, the layer selection protocol, the evaluation prompts, and the judge validation (Appendix B). This is the primary reference for how to execute this kind of project well. arXiv: 2507.21509
- 4. Tan, Hazineh, Zhou et al. 2024 "Analysing the Generalisation and Reliability of Steering Vectors." Critical for calibrating expectations. Shows that steering vectors often fail to generalize, that steerability is a dataset-level property, and that many behaviors have substantial "anti-steerable" examples. This directly affects how we interpret composition failures: if a pair fails to compose, is it because of geometric interference (our hypothesis) or because one of the behaviors is dataset-unsteerable to begin with (their finding)? We must be able to distinguish these.

arXiv: 2407.12404

5. Zou et al. 2023 — "Representation Engineering: A Top-Down Approach to AI Transparency." The broader framework (RepE) in which steering vectors sit. Introduces PCA-based direction extraction and the general view of behaviors as linear directions. Read the first third carefully (conceptual framing and the LAT/RepE method); skim the rest as a catalogue. Needed for situating our work in the introduction and related-work sections. arXiv: 2310.01405

# Tier 2 — Split across the team, Weeks 1–2

Each person takes two or three papers from their category, writes a one-paragraph summary, and shares it in a collective related\_work.md file. These summaries seed the related-work section of the final writeup.

##### For the Analysis Lead (geometry, composition, statistical methods)

6. Im and Li 2025 — "A Unified Understanding and Evaluation of Steering Methods." Theoretical framework unifying mean-difference, PCA, and logistic-regression constructions of steering vectors. Argues mean-difference is near-optimal under mild assumptions. Cite when justifying our methodological choice.

arXiv: 2502.02716

- 7. Postmus et al. 2024 "From Steering Vectors to Conceptors: Compositional Affine Activation Steering for LLMs." The most thoughtful recent attempt at principled multi-behavior steering, using conceptor theory to compose behaviors via Boolean operations on soft projection matrices. Directly adjacent to our composition work. Important to cite because it is the strongest existing alternative to the additive-composition approach we study. Read the composition section carefully; the conceptor math can be skimmed. OpenReview: 0Yu0eNdHyV
- 8. Steer2Adapt (2026). Builds a reusable low-dimensional subspace for behaviors and composes basis vectors for new tasks via Bayesian optimization on coefficients. The most similar recent work in spirit, though it takes a different angle (optimization-based composition rather than geometric-prediction). Must be read carefully enough to distinguish our contribution in the related-work section.

arXiv: 2602.07276 (verify on arXiv)

##### For the Evaluation Lead (judge protocols, behavior scoring, datasets)

9. Perez et al. 2022 — "Discovering Language Model Behaviors with Model-Written Evaluations." The source of our contrastive datasets. Read the construction methodology so the team understands exactly what is in the data and where its known biases lie. Note that MWE datasets are LLM-generated and may have subtle artifacts that affect what our steering vectors actually capture.

arXiv: 2212.09251

- 10. Zheng et al. 2023 "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." The canonical reference for LLM-as-judge methodology. Read for the known failure modes — position bias, verbosity bias, self-preference bias — and the standard mitigations. Our judge protocol must address these. Cite when describing our evaluation protocol. arXiv: 2306.05685
- 11. Chen et al. 2025 Persona Vectors, Appendix B. Re-read this appendix specifically. The judge validation protocol (30 human-labeled examples, Spearman agreement, iteration on prompt until threshold met) is the template for our Week 1 judge validation.
- 12. Panickssery et al. 2024 Evaluation section. Re-read the evaluation section of the CAA paper. The multiple-choice and open-ended generation protocols are the templates we will adapt.

##### For the Infrastructure Lead (tooling, hooks, model internals)

- 13. TransformerLens documentation and tutorials (Nanda & Bloom). Not a paper, but required. Spend a full day working through the official tutorials before writing the extraction pipeline. Will save a week of debugging. Pay particular attention to the HookedTransformer class, the run\_with\_cache method, and how to identify the residual-stream hook points (blocks.L.hook\_resid\_post and similar).
  - URL: <https://github.com/TransformerLensOrg/TransformerLens>
- 14. Elhage et al. 2021 "A Mathematical Framework for Transformer Circuits" (Anthropic). Skim sections 1–3. Gives the vocabulary — residual stream, read/write weights, layer norm — needed to explain where and why we extract activations. URL: <https://transformer-circuits.pub/2021/framework>
- 15. Llama 3.1 model card and architecture details. Read the official model card for meta-llama/Llama-3.1-8B and Llama-3.1-8B-Instruct. Confirm the hidden dimension

(4096), number of layers (32), attention head dimensions, and any tokenizer specifics. Know the model before extracting from it.

##### For the Bridge/Mechanism Lead (mechanism analysis, per-layer projections, base-vs-instruct)

- 16. Arditi et al. 2024 "Refusal in Language Models Is Mediated by a Single Direction." Read carefully, not as background. This is the structural template for RQ2's mechanism analysis: extraction of a direction, projection of activations across layers, interpretation of per-layer dynamics, and directional ablation as a causal check. Bri's pipeline resembles theirs. Pay particular attention to how they present per-layer projection results and how they argue for a single-direction interpretation. arXiv: 2406.11717
- 17. Elhage et al. 2021 "A Mathematical Framework for Transformer Circuits" (Anthropic). Same paper as Infrastructure Lead's Tier 2, but Bri reads sections 1–2 in depth, focusing on the residual stream as a communication channel and on how attention and MLP components read from and write to it linearly. This is the conceptual justification for why projecting downstream activations onto v (L⋆) is a meaningful mechanistic operation. URL: <https://transformer-circuits.pub/2021/framework>
- 18. Chen et al. 2025 Persona Vectors (full paper, re-read). Already Tier 1, but Bri re-reads sections 3 and 4 closely for the base-vs-instruct observations and the per-layer persona vector tracking. This is the closest published work to RQ2; Bri needs to be able to articulate precisely how our mechanism framing differs from theirs (per-layer projection trajectories under joint steering vs. per-layer vector extraction and monitoring). arXiv: 2507.21509
- 19. Geva et al. 2023 "Dissecting Recall of Factual Associations in Auto-Regressive Language Models." The methodological reference for per-layer projection-based analysis. Carefully works through how information flows through the residual stream during factual recall, using exactly the projection-based techniques we are adapting. Read for methodology, not for content. Will save time when structuring the trajectory analysis. arXiv: 2304.14767
- 20. nostalgebraist 2020 / Belrose et al. 2023 "Logit Lens" and "Tuned Lens." Both are directly relevant as examples of per-layer projection onto a fixed basis (the unembedding). Belrose et al. in particular gives a careful treatment of when per-layer projection is informative and when it misleads. Read Belrose et al. in Week 2; blog post on Logit Lens can be a 30-minute background read. arXiv: 2303.08112 (Belrose et al.)
- 21. Anthropic 2026 "The Assistant Axis: Situating and Stabilizing the Character of Large Language Models." Recent (January 2026). Claims many persona-like behaviors share a single underlying axis, built by post-training. Directly relevant to Part C: if our base-vs-instruct mechanism comparison shows systematic differences, Assistant Axis is the natural theoretical frame. If it shows no difference, Assistant Axis is a point of tension worth addressing.

URL: <https://www.anthropic.com/research/assistant-axis>

### Tier 3 — Background and comparison, consult during writeup

Do not read front-to-back. Consult tactically for the related-work section and to anticipate reviewer questions.

22. Turner et al. 2023 — "Activation Addition: Steering Language Models Without Optimization." Early and influential work on activation engineering. Cite as a foundational method paper.

arXiv: 2308.10248

- 23. Li et al. 2024 "Inference-Time Intervention (ITI): Eliciting Truthful Answers from a Language Model." Alternative construction method using probing directions. Cite when justifying mean-difference over probe-based alternatives. arXiv: 2306.03341
- 24. Valence–Arousal Subspace paper (2026). The emotion-specific version of the lowdimensional-behavior-subspace question. Read to ensure correct citation and to position our non-emotion focus against their emotion focus. arXiv: 2604.03147 (verify on arXiv)
- 25. Park, Choe, Veitch 2024 "The Linear Representation Hypothesis and the Geometry of Large Language Models." Theoretical grounding for why linear subspaces are the right object to study in the first place. Cite in the introduction when justifying the whole framing.

arXiv: 2311.03658

- 26. Wehner et al. 2025 "Representation Engineering: A Taxonomic Survey." Recent comprehensive survey. Useful during writeup for finding references we may have missed and for situating our work in the broader RepE landscape. arXiv: 2502.19649
- 27. Vu et al. 2025/2026 "Curveball Steering: The Right Direction To Steer Isn't Always Linear." Challenges linear-subspace assumptions using polynomial kernel PCA. Cite as a methodological limitation of our linear approach. If our logistic regression AUC is disappointing, this paper offers a hypothesis for why (the true geometry may not be linear). arXiv: 2603.09313 (verify on arXiv)
- 28. He et al. 2025 "SAE-SSV: Supervised Steering in Sparse Representation Spaces for Reliable Control of Language Models." Alternative approach using sparse autoencoders to constrain steering to interpretable subspaces. Cite as a parallel research direction — different philosophy (disentangle via SAE) but same ultimate goal (clean multibehavior control).

arXiv: 2505.16188

29. Templeton et al. 2024 (Anthropic) — "Scaling Monosemanticity." SAE-based feature analysis as an alternative to linear direction analysis. Bri should be able to articulate in the writeup why the paper uses the latter rather than the former.

URL: <https://transformer-circuits.pub/2024/scaling-monosemanticity>

##### Reading Schedule

Week 1, Day 1, morning (2 hours). All four read Tier 1 papers #1 (CAA) and #2 (Weij et al., composition) together. Brief discussion: what is CAA, and why does naive composition fail?

Week 1, Day 1, afternoon (3 hours). Split the remaining Tier 1 papers: one person each for #3, #4, #5 (with Bri doubling up on #3 since Persona Vectors is directly relevant to Part B and C). Each presents for 15 minutes at the end of the day. Collective discussion: how does our project fit relative to these?

Week 1, Days 2–5 (background reading during infrastructure work). Each person reads their Tier 2 assignments while building their pipeline. Target: one Tier 2 paper per day, summarized in a shared related\_work.md file as a one-paragraph note with the specific claim to be cited. Bri's Tier 2 list is on the heavier side (6 papers); if reading budget is tight, the non-negotiable core is Arditi, Elhage, Persona Vectors re-read, and Geva — the rest can slip to Week 2.

Week 1, end-of-week synchronization (1 hour). Go through related\_work.md together. Identify gaps. Discuss whether any Tier 3 papers have become more relevant than expected.

Weeks 2–3. No mandatory new reading during execution. Keep a running log of any new arXiv postings that look relevant in a shared channel.

Week 4. Consult Tier 3 during writeup. Do not read exhaustively; use tactically — when a citation for "linear representation hypothesis" is needed, pull Park et al.; when a reviewer might ask "why linear?", pull Curveball; when a reviewer asks "why not SAE features?", pull Templeton.

# Monitoring Concurrent Work

This field moves weekly. Between now and submission, it is likely that one or two arXiv papers will appear that touch this project. One team member checks arxiv.org/list/cs.LG/new and cs.CL/new every Monday, filtering for "steering," "activation steering," "representation engineering," "composition," "persona," "mechanistic interpretability," and "circuit." Anything relevant is posted to the shared channel the same day.

If a paper appears in Weeks 2 or 3 that overlaps directly with our Gram-matrix comparison, composition-prediction experiment, or per-layer projection-based mechanism analysis, it must be surfaced to the team immediately. Response options at that point are: (i) reposition the contribution as confirmatory/methodological improvement, (ii) pivot emphasis to the lessoverlapping component, (iii) extend scope in a direction the new paper does not cover (for example, add the sign-asymmetric coefficient analysis or a second model family). Catching this early preserves novelty; catching it at writeup is painful.

A specific watchlist: the Anthropic interpretability team (especially the persona vectors authors), the CAA group, Neel Nanda's students, and the teams behind Steer2Adapt and SAE-SSV. These are the groups most likely to publish something directly adjacent.