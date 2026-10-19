# Concept coherence and semantic alignment

## Files

- `concept_list_n500_main_analysis_clean.csv`: 945 concepts, lexical controls, contextual coherence, three representational-coherence measures, and mean cross-language alignment.
- `concept_centroids_n500_three_models.npz`: raw concept centroids from RoBERTa-base (layer 12; 768 dimensions), GPT-2 large (layer 36; 1,280 dimensions), and Mistral-7B-v0.3 (layer 32; 4,096 dimensions).
- `analysis.py`: concept regressions, pairwise correlations, coherence–concreteness interactions, simple slopes, and domain-dimensionality analyses.

Missing values are retained. Contextual coherence is the negative of Hoffman et al.'s semantic-diversity score; representational coherence is mean pairwise cosine similarity across a concept's contextual representations.

## Run

Tested with Python 3.10. Install the packages in `requirements.txt`, then run:

```bash
python analysis.py
```

Results are written to `results/` beside the script. Input and output paths can be supplied explicitly:

```bash
python analysis.py --concepts concept_list_n500_main_analysis_clean.csv \
  --centroids concept_centroids_n500_three_models.npz --output results \
  --resamples 1000 --seed 20261007
```

Use `--concept-only` to omit analyses requiring the centroid file. The script does not download models or require a GPU. Output tables contain coefficients, standard errors, confidence intervals, sample sizes, nested-model comparisons, and correlation estimates; figures are not rendered by this script.

## Concept analyses

The baseline contains SUBTLEX-US Zipf frequency, word length, and categorical POS. Contextual coherence and each model's representational coherence are added separately. Unique effects compare a joint model containing contextual and representational coherence against the corresponding single-coherence model. All comparisons use the same 933 complete cases; concreteness is not a baseline covariate in this three-model analysis.

Interaction models include frequency, length, POS, concreteness, one coherence measure, and their coherence × concreteness interaction. They use a shared sample of 929 concepts. Contextual and representational coherence enter separate interaction models. Simple slopes evaluate concreteness at coherence values of −1, 0, and +1 SD; these are model probes, not separately fitted subgroups. Simple-slope intervals use ±1.96 SE. Prediction tables hold continuous controls at their means and POS at the reference category (adjectives).

Continuous variables and the outcome are standardized within the relevant complete-case sample, using population SD (`ddof=0`). POS codes are read from the suffix of `Concept_ID`: the descriptive POS gloss combines two categories distinguished in the regression. Correlations use pairwise deletion: N = 933 when contextual coherence is involved and N = 945 otherwise. Reported p-values are two-sided and unadjusted for multiple comparisons.

## Centroid file

Load without pickle:

```python
import numpy as np

with np.load('concept_centroids_n500_three_models.npz', allow_pickle=False) as data:
    concept_ids = data['Concept_ID']
    roberta = data['roberta_centroids']  # 945 × 768
    gpt2 = data['gpt2_centroids']        # 945 × 1280
    mistral = data['mistral_centroids']  # 945 × 4096
```

Rows follow the concept CSV; the script also aligns and checks IDs and domain labels. The archive includes concept names, domains, model identifiers, layer numbers, hidden sizes, and context counts. A centroid is the arithmetic mean of raw target-word representations across sampled Wikipedia contexts. Target subtokens were mean-pooled. There are 500 contexts per concept except `convey`, which has 499. Embedding dimensions are model-specific; columns should not be aligned or pooled across models.

## Domain analyses

Each domain is represented by its member concepts' centroids. Effective dimensionality is the participation ratio, `(sum eigenvalues)^2 / sum(eigenvalues^2)`, after row L2 normalization and column centering. PCA k90 is the smallest number of components explaining at least 90% of variance in column-centered raw centroids, without row normalization. These preprocessing choices differ intentionally.

Domain outcomes are unweighted means of concept alignment. Partial Pearson correlations control the number of concepts in the domain. Parametric tests use `df = n_domains − 3` (18 for 21 domains). The original residual-correlation implementation used an unadjusted Pearson-test df; this script uses the covariate-adjusted df. Residual-permutation p-values and percentile domain-bootstrap intervals use 1,000 resamples by default. The same domain ordering and seed reproduce the original effective-dimensionality resampling results up to numerical precision. Centroid aggregation and SVD precision can produce small floating-point differences in effective-dimensionality values. Domain k90 is strongly associated with domain size; linear adjustment does not establish a sample-size-independent intrinsic dimension.

The supplied inputs support analyses of the saved concept metrics and centroid geometry. They do not contain individual contextual embeddings: concept coherence cannot be independently recalculated, and contextual point-cloud illustrations cannot be reconstructed. No text sampling, embedding extraction, BERT/DeBERTa comparisons, or mixed-effects analyses are included.
