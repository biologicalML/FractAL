"""
Significance testing utilities for nested (split x seed) AL results.

Both functions take:
    a, b : np.ndarray of shape (n_splits, n_seeds)
        Performance of method A and method B, with matching split/seed
        indices (i.e., a[i, j] and b[i, j] come from the same split i and
        the same seed j, so within-split pairing is meaningful).

Higher-is-better metrics are assumed by default (diff = a - b); pass
higher_is_better=False for MSE-style metrics to flip the sign so that a
positive result always means "a beats b".
"""

import numpy as np
from scipy import stats


def paired_split_test(a, b, higher_is_better=True, test="ttest"):
    """
    (1) Aggregate seeds within each split, then run a paired test on the
    resulting n_splits paired values. This treats splits as the unit of
    independent replication and seeds as pseudo-replicates within a split.

    Parameters
    ----------
    a, b : ndarray, shape (n_splits, n_seeds)
    higher_is_better : bool
    test : {"ttest", "wilcoxon"}
        "ttest"    -> paired Student's t-test (assumes normality of the
                      per-split mean differences)
        "wilcoxon" -> Wilcoxon signed-rank test (distribution-free,
                      recommended when n_splits is small, e.g. <= 5-10)

    Returns
    -------
    dict with keys:
        n_splits, mean_diff, statistic, p_value, per_split_diff
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, "a and b must have the same shape"

    # average over seeds within each split -> one value per split
    a_split = a.mean(axis=1)
    b_split = b.mean(axis=1)

    diff = a_split - b_split
    if not higher_is_better:
        diff = -diff

    if test == "ttest":
        stat, p = stats.ttest_rel(a_split, b_split)
        if not higher_is_better:
            stat = -stat
    elif test == "wilcoxon":
        try:
            stat, p = stats.wilcoxon(a_split, b_split)
        except ValueError as e:
            # e.g. all differences are zero
            stat, p = np.nan, np.nan
    else:
        raise ValueError(f"Unknown test: {test}")

    return {
        "n_splits": a.shape[0],
        "mean_diff": float(diff.mean()),
        "statistic": float(stat),
        "p_value": float(p),
        "per_split_diff": diff,
    }


def hierarchical_bootstrap_test(
    a, b, higher_is_better=True, n_boot=10000, alpha=0.05, seed=0
):
    """
    (2) Hierarchical (two-stage cluster) bootstrap. Resamples splits with
    replacement, and within each resampled split resamples seeds with
    replacement, recomputing mean(a) - mean(b) each time. This respects
    both levels of the nested split/seed structure.

    Parameters
    ----------
    a, b : ndarray, shape (n_splits, n_seeds)
    higher_is_better : bool
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Significance level for the two-sided CI (e.g. 0.05 -> 95% CI).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys:
        observed_diff, boot_diffs, ci_low, ci_high, p_value
        (p_value is the two-sided bootstrap p-value: 2 * min(P(diff<=0),
        P(diff>=0)) over the bootstrap distribution, capped at 1)
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, "a and b must have the same shape"
    n_splits, n_seeds = a.shape
    rng = np.random.default_rng(seed)

    def signed(x):
        return x if higher_is_better else -x

    observed_diff = signed(a.mean() - b.mean())

    boot_diffs = np.empty(n_boot)
    for t in range(n_boot):
        split_idx = rng.integers(0, n_splits, size=n_splits)
        a_resampled = np.empty((n_splits, n_seeds))
        b_resampled = np.empty((n_splits, n_seeds))
        for k, s_idx in enumerate(split_idx):
            seed_idx = rng.integers(0, n_seeds, size=n_seeds)
            a_resampled[k] = a[s_idx, seed_idx]
            b_resampled[k] = b[s_idx, seed_idx]
        boot_diffs[t] = signed(a_resampled.mean() - b_resampled.mean())

    ci_low, ci_high = np.percentile(
        boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    p_low = np.mean(boot_diffs <= 0)
    p_high = np.mean(boot_diffs >= 0)
    p_value = float(min(1.0, 2 * min(p_low, p_high)))

    return {
        "observed_diff": float(observed_diff),
        "boot_diffs": boot_diffs,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": p_value,
    }


def one_sample_split_test(
    a, chance, higher_is_better=True, test="ttest", alternative="greater"
):
    """
    One-sample analog of paired_split_test: tests whether a single method's
    performance differs from a fixed constant (e.g., the chance-level recall
    |set|/K under a uniform allocation), rather than comparing two methods.

    As with paired_split_test, seeds are first averaged within each split to
    respect the nested split/seed structure, giving n_splits values that are
    then compared against the constant `chance`.

    Parameters
    ----------
    a : ndarray, shape (n_splits, n_seeds)
        Performance of the method being tested (e.g., FractAL's Worst-Set
        Recall on B2).
    chance : float
        The fixed reference value to test against (e.g., |set|/K).
    higher_is_better : bool
        If True, "outperform" means a > chance. If False (e.g. MSE),
        "outperform" means a < chance; internally this flips the sign so
        alternative="greater" still means "the method is better than chance".
    test : {"ttest", "wilcoxon"}
        "ttest"    -> one-sample Student's t-test
        "wilcoxon" -> one-sample Wilcoxon signed-rank test against `chance`
                      (drop the "vs constant" version if scipy < 1.9, see
                      note below)
    alternative : {"two-sided", "greater", "less"}
        "greater" is the natural choice for a claim like "significantly
        outperforms chance" -- it is a directional (one-sided) test. Use
        "two-sided" if you only want to claim the mean differs from chance
        without committing to direction (e.g., you'd also want to flag a
        method that is significantly *worse* than chance, which "greater"
        would report as non-significant regardless of how extreme it is).

    Returns
    -------
    dict with keys:
        n_splits, mean, chance, mean_diff, statistic, p_value, per_split_diff
    """
    a = np.asarray(a, dtype=float)
    a_split = a.mean(axis=1)  # average over seeds within each split

    diff = a_split - chance
    if not higher_is_better:
        diff = -diff
        a_split_signed = -a_split
        chance_signed = -chance
    else:
        a_split_signed = a_split
        chance_signed = chance

    if test == "ttest":
        stat, p = stats.ttest_1samp(
            a_split_signed, chance_signed, alternative=alternative
        )
    elif test == "wilcoxon":
        # one-sample Wilcoxon signed-rank against a constant: shift by the
        # constant and test whether the shifted sample is symmetric about 0
        shifted = a_split_signed - chance_signed
        if np.all(shifted == 0):
            stat, p = np.nan, np.nan
        else:
            stat, p = stats.wilcoxon(shifted, alternative=alternative)
    else:
        raise ValueError(f"Unknown test: {test}")

    return {
        "n_splits": a.shape[0],
        "mean": float(a_split.mean()),
        "chance": float(chance),
        "mean_diff": float(diff.mean()),
        "statistic": float(stat),
        "p_value": float(p),
        "per_split_diff": diff,
    }


def hierarchical_bootstrap_vs_constant(
    a, chance, higher_is_better=True, n_boot=10000, alpha=0.05, seed=0
):
    """
    Hierarchical bootstrap analog for testing a single method against a
    fixed constant (e.g., chance-level recall). Resamples splits with
    replacement, then resamples seeds within each resampled split with
    replacement, and asks what fraction of bootstrap means fall at or
    below `chance` (equivalently, whether the bootstrap CI excludes chance).

    Parameters
    ----------
    a : ndarray, shape (n_splits, n_seeds)
    chance : float
    higher_is_better : bool
    n_boot : int
    alpha : float
        Used for the two-sided CI; the reported p-value is one-sided
        (P(bootstrap mean <= chance), the natural quantity for an
        "outperforms chance" claim).

    Returns
    -------
    dict with keys:
        observed_mean, chance, ci_low, ci_high, p_value_greater
    """
    a = np.asarray(a, dtype=float)
    n_splits, n_seeds = a.shape
    rng = np.random.default_rng(seed)

    def signed(x):
        return x if higher_is_better else -x

    chance_signed = signed(chance)
    observed_mean = signed(a.mean())

    boot_means = np.empty(n_boot)
    for t in range(n_boot):
        split_idx = rng.integers(0, n_splits, size=n_splits)
        a_resampled = np.empty((n_splits, n_seeds))
        for k, s_idx in enumerate(split_idx):
            seed_idx = rng.integers(0, n_seeds, size=n_seeds)
            a_resampled[k] = a[s_idx, seed_idx]
        boot_means[t] = signed(a_resampled.mean())

    ci_low, ci_high = np.percentile(
        boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    # one-sided p-value: probability the bootstrap mean is at or below chance
    p_value_greater = float(np.mean(boot_means <= chance_signed))

    return {
        "observed_mean": float(observed_mean if higher_is_better else -observed_mean),
        "chance": float(chance),
        "ci_low": float(ci_low if higher_is_better else -ci_high),
        "ci_high": float(ci_high if higher_is_better else -ci_low),
        "p_value_greater": p_value_greater,
    }