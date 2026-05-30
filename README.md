# Predicting Behavior Composition from Steering-Vector Geometry

Code for the workshop paper *Predicting Behavior Composition from Steering-Vector Geometry*
(30562 — Machine Learning and Artificial Intelligence, Bocconi University).

**Question.** When two individually steerable behavior directions are added during a forward pass,
does the geometry of the two vectors predict the joint outcome? We study eight persona/style
directions in `Llama-3.1-8B-Instruct`, evaluate all 28 pairs under two controlled additive
composition schemes, and decompose each joint outcome into a **behavioral magnitude** `M` and a
**directional residual** `S` (reinforcement vs. suppression relative to single-vector steering).

**Headline result.** Signed cosine predicts the directional residual under both schemes
(`β = +0.310`, exact node-perm `p = 0.047` for normalized-sum; `β = +0.411`, `p < 0.001` for
projection-controlled), while magnitude has smaller, less-robust coefficients (`β ≈ 0.19`).

---

## Repository layout

```
src/                         Importable library (paper pipeline)
  judge.py                   OpenAI log-prob judge (0–100 probability-weighted score)
  scoring.py                 Per-behavior judge rubrics + judge model id
  datasets.py                Contrastive-pair / MWE loaders, deterministic split
  extraction/                Persona-vector extraction (Anthropic-replication pipeline)
    trait_data.py            Trait-artifact loader + system-prompt builder
    generation.py            Chat-template generation + async trait/coherence judging
    build_vector.py          Difference-of-means direction per layer (response-averaged)
  inference/
    hf_model.py              HF model loader + response-token steering hook
    hf_logprob.py            Multiple-choice log-prob delta under steering
  geometry/
    clusters.py              Behavior-cluster covariate (antisocial vs. other)
    pair_strat.py            Pair table + cosine-stratum assignment
  composition/
    joint_injection.py       compose_steering_vector (normalize: False/True/per_axis) + batched steering
    joint_behaviors.py       Pair enumeration
    joint_judge.py           Per-row two-behavior + coherence judging
    human_samples.py         Joint-steered completion sampler for the human-eval sheet

scripts/                     Runnable drivers (one stage each)
  extraction/                Trait-artifact + MWE generation, extract, build vectors
  validation/                Single-vector dual-signal validation @ L=17, α-sweep, norm diagnostic
  layer_selection/           Per-trait + shared-layer sweep (selects L*=17)
  compositions/              The two composition schemes + judging + aggregation + α calibration
  semantics/                 Behavior-description semantic-similarity control
  human_eval/                Human vs. judge agreement (Appendix D)

analysis/
  rq1/                       RQ1 headline: exact 8! node-permutation (MRQAP) regression
  rq1_consolidated/          Frozen analysis frame (one row per pair × scheme, coh-gated)
  results/RQ1/               Reproduced RQ1 report (.md + .json)

data/behaviors_mwe/          Multiple-choice datasets for the 9 validated traits (log-prob check)
external/anthropic_code/     Vendored Persona Vectors trait artifacts + prompt template (extraction inputs)
slurm/                       Cluster job wrappers for each driver
docs/                        Research plan + method explainers
```

---

## The pipeline (and which script runs each stage)

| Stage | Script | Output |
|---|---|---|
| 1. Trait artifacts | `scripts/extraction/generate_trait_artifacts.py` | trait JSONs for project-generated behaviors |
| 2. Extract + judge | `scripts/extraction/run_extract.py`, `run_extract_all.py` | 500 pos / 500 neg generations per trait, judged |
| 3. Build vectors | `scripts/extraction/run_build_vector.py` | per-layer difference-of-means directions `[33, 4096]` |
| 4. Layer selection | `scripts/layer_selection/run_layer_selection_all.py` | shared `L*=17` |
| 5. Single-vector validation | `scripts/validation/run_validation_all_layer17.py`, `run_alpha_sweep_l17.py`, `check_norms_layer17.py` | judge + log-prob dual-signal; α=4 single-vector operating point |
| 6. MWE datasets | `scripts/extraction/generate_mwe_behaviors.py` | log-prob check data in `data/behaviors_mwe/` |
| 7. Composition — normalized-sum | `scripts/compositions/composition_scoring_v2.py` (`normalize=True`, α=4.5) | judged completions for 28 pairs × 4 settings |
| 7. Composition — projection-controlled | `scripts/compositions/composition_scoring_v3.py` (`per_axis`, α=4.5) | same, per-axis injection |
| 7b. Composition α calibration | `scripts/compositions/validate_normalisations_pilot*.py` | selects α=4.5 (Appendix C) |
| 8. Judge + aggregate | `scripts/compositions/composition_judge_local.py`, `composition_aggregate_local.py` | per-pair Δ summary |
| 9. Semantic control | `scripts/semantics/semantic_sim.py` | `sem_sim` covariate |
| 10. Judge validation | `scripts/human_eval/*` | human vs. judge agreement (ρ=0.89) |
| 11. **RQ1 headline** | `analysis/rq1/nodeperm_analysis.py` | the paper's main + robustness tables |

`scripts/compositions/composition_scoring.py` is the shared driver (and the unnormalized-sum
baseline of Appendix F, kept as the import base for the v2/v3 schemes).

---

## Reproduce the headline result (no GPU, no API)

The analysis frame is shipped frozen, so the paper's main statistics reproduce directly:

```bash
cd analysis/rq1
python nodeperm_analysis.py
```

Reads `analysis/rq1_consolidated/consolidated_coh30.csv`, fits `y ~ mean_single_abs + max_single_abs + cos`
per scheme, and recomputes `β_cos`'s `p`-value with the exhaustive `8! = 40,320` node-permutation null.
Writes `analysis/results/RQ1/perscheme_nodeperm_analysis/{md,json}` — the directional-residual,
magnitude, `|cos|`, `+sem_sim`, and leave-one-trait-out tables of the paper.

Requires only `numpy`, `pandas`, `scipy` (from `requirements.txt`).

---

## Running the full pipeline

The generation/extraction/composition stages need a GPU (`Llama-3.1-8B-Instruct`, bf16) and an
`OPENAI_API_KEY` for the `gpt-4.1-mini` judge. Put the key in a `.env` at the repo root. See
[README environment notes below](#environment) and the `slurm/` wrappers for cluster launch.
Each driver is idempotent (skip-if-output-exists) and is run from the repo root, e.g.
`python -m scripts.compositions.composition_scoring_v2`.

### Environment

Two environments share a core set of pinned packages but install PyTorch differently:

| File | Purpose |
|---|---|
| `requirements.txt` | Core deps, identical across machines |
| `requirements-mac.txt` | CPU-only PyTorch for local development/analysis |
| `requirements-hpc.txt` | GPU/quantization support for the cluster |

```bash
python3.11 -m venv venv && source venv/bin/activate
pip install -U pip
pip install -r requirements-mac.txt -r requirements.txt   # local
# on the cluster: install CUDA-matched torch, then -r requirements-hpc.txt -r requirements.txt
```

---

## Data and attribution

Contrastive trait artifacts for the six released behaviors (`apathetic`, `evil`, `hallucinating`,
`humorous`, `impolite`, `sycophantic`) are adapted from the **Persona Vectors** release:

> Chen, Arditi, Sleight, Evans, Lindsey (2025). *Persona Vectors: Monitoring and Controlling
> Character Traits in Language Models*. Anthropic. https://github.com/safety-research/persona_vectors

Artifacts for `confidence` and `formality` were generated with the same prompt template
(`external/anthropic_code/data_generation/prompts.py`) using `gpt-4.1`. The vendored Anthropic
prompt template and trait-data JSONs live under `external/anthropic_code/` and are inputs to the
extraction pipeline.
