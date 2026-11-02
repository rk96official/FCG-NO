import numpy as np
from scipy import stats

from fcgno.eval.metrics import iterations_to_tolerance


# Empirical quantile that never interpolates, so failed systems (inf) stay infinite
def quantile(values, level):
    return float(np.quantile(values, level, method="inverted_cdf"))


# JSON-safe number: inf and nan become None
def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


# Iterations to tol per system; systems that never reach tol get inf
def per_system_iterations(relative_residual, tol):
    counts = iterations_to_tolerance(relative_residual, tol).astype(float)
    counts[counts < 0] = np.inf
    return counts


# Summary over systems: failure rate, median, p90, max (None when failures make them undefined) and the paper's mean-curve count
def summarize_systems(relative_residual, tol):
    counts = per_system_iterations(relative_residual, tol)
    return {
        "systems": int(len(counts)),
        "failure_rate": float(np.mean(~np.isfinite(counts))),
        "median": finite_or_none(quantile(counts, 0.5)),
        "p90": finite_or_none(quantile(counts, 0.9)),
        "max": finite_or_none(counts.max()),
        "mean_curve": int(iterations_to_tolerance(np.mean(relative_residual, axis=0), tol)),
    }


# Percentile bootstrap confidence interval of a statistic of per-system counts (pooled over seeds if desired)
def bootstrap_interval(counts, statistic, level=0.95, n_resamples=2000, seed=0):
    samples = np.random.default_rng(seed).choice(counts, size=(n_resamples, len(counts)), replace=True)
    values = np.array([statistic(sample) for sample in samples])
    lower, upper = np.quantile(values, [(1 - level) / 2, (1 + level) / 2], method="inverted_cdf")
    return finite_or_none(lower), finite_or_none(upper)


# Area under the ROC curve of scores for binary labels (Mann-Whitney U / (n1 n0)); None when a class is empty
def roc_auc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    positives, negatives = scores[labels], scores[~labels]
    if len(positives) == 0 or len(negatives) == 0:
        return None
    return float(stats.mannwhitneyu(positives, negatives).statistic / (len(positives) * len(negatives)))


# Paired Wilcoxon signed-rank p-value of per-system counts, with failures set to the iteration cap + 1
def paired_wilcoxon(counts_a, counts_b, cap):
    first = np.where(np.isfinite(counts_a), counts_a, cap + 1)
    second = np.where(np.isfinite(counts_b), counts_b, cap + 1)
    if np.all(first == second):
        return 1.0
    return float(stats.wilcoxon(first, second).pvalue)
