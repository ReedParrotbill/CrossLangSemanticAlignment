"""Concept coherence and domain dimensionality analyses.

Inputs: a concept-level CSV and an NPZ archive of three-model centroids.
Analyses: OLS regressions, pairwise correlations, concreteness interactions,
and domain-size-adjusted dimensionality correlations.

Run: python analysis.py --output results
Default input paths are relative to this script. Tables are saved as CSV files.
See README.md for variable definitions and package requirements.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import scipy
from scipy import linalg, stats
import statsmodels
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
MODELS = {'RoBERTa': 'roberta_centroids', 'GPT2': 'gpt2_centroids',
          'Mistral': 'mistral_centroids'}
COLUMNS = {
    'Concept_ID': 'id', 'Concept': 'word', 'Domain': 'domain',
    'Word frequency (Brysbaert & New, 2009; SUBTLEX-US Zipf)': 'frequency',
    'Word length': 'length',
    'Contextual coherence (Hoffman et al., 2013; -SD)': 'CC',
    'Concreteness (Brysbaert et al., 2014)': 'concreteness',
    'Cross-language alignment (Thompson et al., 2020; mean)': 'alignment',
    **{f'Representational coherence_{m}': m for m in MODELS},
}


def standardize(x):
    """Z-score using the sample mean and population SD (ddof=0)."""
    return (x - x.mean()) / x.std(ddof=0)


def fit_ols(data, predictors, interaction=None):
    """Fit standardized OLS with categorical POS and an optional product term."""
    x = pd.DataFrame({v: standardize(data[v]) for v in predictors})
    if interaction:
        a, b = interaction
        x[f'{a}:concreteness'] = x[a] * x[b]
    pos = pd.get_dummies(data['pos'], prefix='POS', drop_first=True, dtype=float)
    x = sm.add_constant(pd.concat([x, pos], axis=1), has_constant='add')
    return sm.OLS(standardize(data.alignment), x.astype(float)).fit()


def coefficients(fit, model):
    """Collect coefficients, standard errors, t tests, and 95% intervals."""
    ci = fit.conf_int()
    return pd.DataFrame({'model': model, 'term': fit.params.index,
                         'beta': fit.params.values, 'SE': fit.bse.values,
                         't': fit.tvalues.values, 'p': fit.pvalues.values,
                         'ci_low': ci[0].values, 'ci_high': ci[1].values,
                         'n': int(fit.nobs), 'r2': fit.rsquared})


def concept_regression(data, out):
    """Estimate additional and unique coherence effects on shared complete cases.

    Baseline: frequency, length, and POS. Additional effects add one coherence
    predictor; unique effects remove one predictor from a joint CC–RC model.
    The supplied data yield 933 concepts.
    """
    required = ['alignment', 'frequency', 'length', 'pos', 'CC', *MODELS]
    d = data.dropna(subset=required).copy()
    base = ['frequency', 'length']
    fitted = {'baseline': fit_ols(d, base)}
    for v in ['CC', *MODELS]:
        fitted[v] = fit_ols(d, base + [v])
    for m in MODELS:
        fitted[f'CC+{m}'] = fit_ols(d, base + ['CC', m])
    rows = []
    for predictor in ['CC', *MODELS]:
        rows.append(('additional', predictor, predictor, 'baseline'))
    for m in MODELS:
        rows.extend([('unique', 'CC', f'CC+{m}', m),
                     ('unique', m, f'CC+{m}', 'CC')])
    effects = []
    for kind, term, full_name, reduced_name in rows:
        full, reduced = fitted[full_name], fitted[reduced_name]
        f, p, df = full.compare_f_test(reduced)
        assert np.isclose(f, full.tvalues[term] ** 2)
        assert np.isclose(full.aic - reduced.aic,
                          full.nobs * np.log((1-full.rsquared)/(1-reduced.rsquared)) + 2)
        effects.append(dict(effect=kind, predictor=term, full_model=full_name,
                            reduced_model=reduced_name, n=int(full.nobs),
                            delta_r2=full.rsquared-reduced.rsquared,
                            delta_aic=full.aic-reduced.aic, beta=full.params[term],
                            F=f, df_num=df, df_resid=full.df_resid, p=p))
    pd.DataFrame(effects).to_csv(out/'concept_additional_unique_effects.csv', index=False)
    pd.concat([coefficients(fit, name) for name, fit in fitted.items()],
              ignore_index=True).to_csv(out/'concept_regression_coefficients.csv', index=False)
    pd.DataFrame([dict(model=name, n=int(fit.nobs), r2=fit.rsquared,
                       adjusted_r2=fit.rsquared_adj, aic=fit.aic)
                  for name, fit in fitted.items()]).to_csv(out/'concept_regression_models.csv', index=False)
    d.to_csv(out/'concept_regression_sample.csv', index=False)


def correlations(data, out):
    """Compute pairwise correlations with pairwise deletion (N=933 or 945)."""
    variables = ['CC', *MODELS, 'alignment']
    rows = []
    for i, a in enumerate(variables):
        for b in variables[i+1:]:
            d = data[[a, b]].dropna()
            r, p = stats.spearmanr(d[a], d[b])
            rows.append(dict(variable_a=a, variable_b=b, n=len(d),
                             spearman_rho=r, spearman_p=p,
                             pearson_r=stats.pearsonr(d[a], d[b]).statistic))
    pd.DataFrame(rows).to_csv(out/'concept_correlations.csv', index=False)


def interactions(data, out):
    """Fit coherence × concreteness models on 929 shared complete cases.

    Covariates are frequency, length, and POS. Simple slopes are evaluated at
    coherence z-scores of -1, 0, and +1, with normal-approximation 95% intervals.
    Predictions hold continuous controls at zero and POS at its reference level.
    """
    d = data.dropna(subset=['alignment', 'frequency', 'length', 'pos', 'CC',
                           'concreteness', *MODELS]).copy()
    coefs, slopes, predictions = [], [], []
    for moderator in ['CC', *MODELS]:
        fit = fit_ols(d, ['frequency', 'length', 'concreteness', moderator],
                      interaction=(moderator, 'concreteness'))
        coefs.append(coefficients(fit, moderator))
        term = f'{moderator}:concreteness'
        cov = fit.cov_params()
        for value in [-1., 0., 1.]:
            slope = fit.params.concreteness + value * fit.params[term]
            variance = (cov.loc['concreteness', 'concreteness']
                        + value**2 * cov.loc[term, term]
                        + 2*value * cov.loc['concreteness', term])
            se = np.sqrt(max(0., variance))
            slopes.append(dict(moderator=moderator, moderator_z=value,
                               n=int(fit.nobs), slope=slope, SE=se,
                               ci_low=slope-1.96*se, ci_high=slope+1.96*se))
            for conc in np.linspace(-2, 2, 81):
                x = pd.DataFrame(0., index=[0], columns=fit.params.index)
                x['const'] = 1.; x['concreteness'] = conc
                x[moderator] = value; x[term] = value*conc
                pred = fit.get_prediction(x).summary_frame().iloc[0]
                predictions.append(dict(moderator=moderator, moderator_z=value,
                                        concreteness_z=conc, predicted_alignment=pred['mean'],
                                        ci_low=pred.mean_ci_lower, ci_high=pred.mean_ci_upper))
    pd.concat(coefs, ignore_index=True).to_csv(out/'interaction_coefficients.csv', index=False)
    pd.DataFrame(slopes).to_csv(out/'concreteness_simple_slopes.csv', index=False)
    pd.DataFrame(predictions).to_csv(out/'interaction_predictions.csv', index=False)
    d.to_csv(out/'interaction_sample.csv', index=False)


def residualize(x, size):
    """Remove the linear association with domain size, including an intercept."""
    c = np.column_stack([np.ones(len(size)), size])
    return x - c @ np.linalg.lstsq(c, x, rcond=None)[0]


def partial_correlation(x, y, size):
    """Pearson correlation between size-adjusted residuals."""
    return float(stats.pearsonr(residualize(x, size), residualize(y, size)).statistic)


def domain_analysis(data, centroid_file, out, resamples, seed):
    """Calculate domain geometry and its association with mean alignment.

    Effective dimensionality uses row-normalized, centered centroids; PCA k90
    uses centered raw centroids. Partial correlations control domain size
    (df=18 for 21 domains), with residual permutations and domain bootstraps.
    """
    rows = []
    with np.load(centroid_file, allow_pickle=False) as z:
        ids = pd.Index(z['Concept_ID'].astype(str))
        if ids.has_duplicates or set(ids) != set(data.id):
            raise ValueError('Centroid IDs must match the concept table exactly.')
        order = ids.get_indexer(data.id)
        if not np.array_equal(z['Domain'][order].astype(str), data.domain.to_numpy(str)):
            raise ValueError('Domain labels do not match the concept table.')
        for model, key in MODELS.items():
            vectors = z[key][order].astype(float)
            if not np.isfinite(vectors).all():
                raise ValueError(f'Non-finite centroids: {model}')
            for domain, g in data.groupby('domain', sort=True):
                x = vectors[g.index]
                # Participation ratio follows row L2 normalization and centering.
                norms = np.linalg.norm(x, axis=1, keepdims=True)
                if np.any(norms == 0): raise ValueError('Zero-norm centroid.')
                unit = x/norms
                ev = linalg.svdvals(unit-unit.mean(0))**2
                effective = ev.sum()**2 / np.square(ev).sum()
                # PCA k90 uses raw centroids, with column centering only.
                raw_ev = linalg.svdvals(x-x.mean(0))**2
                k90 = int(np.searchsorted(np.cumsum(raw_ev)/raw_ev.sum(), .90)+1)
                rows.append(dict(model=model, domain=domain, n_concepts=len(g),
                                 alignment=g.alignment.mean(), effective_dimensionality=effective,
                                 pca_k90=k90))
    domains = pd.DataFrame(rows)
    domains.to_csv(out/'domain_metrics.csv', index=False)
    results = []
    for metric in ['effective_dimensionality', 'pca_k90']:
        rng = np.random.default_rng(seed)
        for model in MODELS:
            d = domains[domains.model.eq(model)].sort_values('alignment', ascending=False)
            y = d.alignment.to_numpy(); size = d.n_concepts.to_numpy(); n = len(d)
            x = d[metric].to_numpy(); r = partial_correlation(x, y, size)
            df = n-3; t = r*np.sqrt(df/(1-r*r))
            rx, ry = residualize(x, size), residualize(y, size)
            # Permute outcome residuals for a two-sided test.
            null = [stats.pearsonr(rx, rng.permutation(ry)).statistic for _ in range(resamples)]
            boot = []
            for _ in range(resamples):
                idx = rng.integers(0, n, n)
                if np.unique(size[idx]).size < 2: continue
                bx, by = residualize(x[idx], size[idx]), residualize(y[idx], size[idx])
                if np.std(bx) > 1e-12 and np.std(by) > 1e-12:
                    boot.append(stats.pearsonr(bx, by).statistic)
            low, high = np.quantile(boot, [.025, .975])
            results.append(dict(model=model, metric=metric, n_domains=n, partial_r=r,
                                t=t, df=df, p=2*stats.t.sf(abs(t), df),
                                permutation_p=(1+np.sum(np.abs(null)>=abs(r)))/(resamples+1),
                                bootstrap_ci_low=low, bootstrap_ci_high=high,
                                n_permutations=resamples, n_bootstraps=len(boot),
                                raw_pearson_r=stats.pearsonr(x,y).statistic,
                                size_pearson_r=stats.pearsonr(x,size).statistic))
    pd.DataFrame(results).to_csv(out/'domain_correlations.csv', index=False)


def main():
    """Read inputs, run analyses, and record package versions and input hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--concepts', type=Path, default=HERE/'concept_list_n500_main_analysis_clean.csv')
    parser.add_argument('--centroids', type=Path, default=HERE/'concept_centroids_n500_three_models.npz')
    parser.add_argument('--concept-only', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE/'results')
    parser.add_argument('--resamples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261007)
    args = parser.parse_args()
    if args.resamples < 1: parser.error('--resamples must be positive')
    data = pd.read_csv(args.concepts).rename(columns=COLUMNS)
    required = list(COLUMNS.values())
    if not set(required).issubset(data): raise ValueError('Missing required concept columns.')
    if data.id.isna().any() or data.id.duplicated().any(): raise ValueError('Invalid concept IDs.')
    # The identifier retains POS distinctions that the human-readable gloss merges.
    data['pos'] = data.id.str.rsplit('::', n=1).str[-1]
    for col in ['alignment', 'frequency', 'length', 'CC', 'concreteness', *MODELS]:
        data[col] = pd.to_numeric(data[col], errors='raise').replace([np.inf,-np.inf],np.nan)
    data = data.reset_index(drop=True)
    args.output.mkdir(parents=True, exist_ok=True)
    concept_regression(data, args.output)
    correlations(data, args.output)
    interactions(data, args.output)
    inputs = [args.concepts]
    if not args.concept_only:
        domain_analysis(data, args.centroids, args.output, args.resamples, args.seed)
        inputs.append(args.centroids)
    metadata = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    scipy=scipy.__version__, statsmodels=statsmodels.__version__,
                    seed=args.seed, resamples=args.resamples,
                    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (args.output/'run_info.json').write_text(json.dumps(metadata, indent=2))
    print(f'Results: {args.output.resolve()}')


if __name__ == '__main__':
    main()
