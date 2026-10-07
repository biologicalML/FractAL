#!/usr/bin/env python3
"""
Compute normalized-advantage gain and rank of adaptive methods, averaged
across datasets.

For each dataset, at the dataset's configured `max_rounds`, per (split,
run) cell:
  - p_rand = the "Random" method's test_perf (comp_meth_folders_dict[data][0])
  - p_best = the best pure strategy's test_perf (max/min over
    pure_strat_folders_dict[data], per higher_is_better)
  - p_i    = an adaptive method's test_perf

Per dataset, p_i/p_rand/p_best are first averaged over all (split, run)
cells, and gain is then computed once from those averaged performances
(a ratio-of-means, not a mean-of-ratios):
  gain(method) = (mean(p_i) - mean(p_rand)) / (mean(p_best) - mean(p_rand))
  (0 = no better than random, 1 = matches the best pure strategy)
This avoids instability from dividing by a near-zero (p_best - p_rand) gap
in individual runs, which can otherwise blow up single-cell ratios into
outliers that dominate a per-cell average.

rank(method) = the method's rank (1 = best) among the other adaptive
methods (excluding Random), computed from each method's dataset-level
mean(p_i) — the same mean used for gain — rather than per-cell, so rank
and gain are derived from the same dataset-level performance numbers.
This gives one integer rank per method per dataset.

Those per-dataset numbers (one gain ratio, one integer rank per method)
are then averaged across all datasets to report:
  - Mean gain / Mean Rank: mean +/- standard error across datasets.
  - Worst gain / Worst Rank: the single worst per-dataset mean (min for
    gain, max for rank) across datasets, no SE.

"Random" is used only as the gain baseline and is not a reported row.
"""
import csv
import json
import os
import numpy as np
import tabulate
import matplotlib.pyplot as plt
import mplscience

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"

LABEL_SIZE = 14
TICK_SIZE = 13

METHOD_COLOR_LIST = {
    "ALBL": "blue",
    "UCB": "purple",
    "SCRiBLe": "orange",
    "SelectAL": "brown",
    "AutoAL": "teal",
    "FractAL": "red",
}


def print_table(results):
    rows = []
    for method, stats in results.items():
        rows.append([
            method,
            f"{stats['mean_gain']:.3f} ± {stats['mean_gain_se']:.3f}",
            f"{stats['median_gain']:.3f}",
            f"{stats['worst_gain']:.3f}",
            f"{stats['mean_rank']:.2f} ± {stats['mean_rank_se']:.2f}",
            f"{stats['median_rank']:.2f}",
            f"{stats['worst_rank']:.2f}",
        ])
    headers = ["Strategy", "Mean gain", "Median gain", "Worst gain", "Mean Rank", "Median Rank", "Worst Rank"]
    print(tabulate.tabulate(rows, headers=headers, tablefmt="fancy_grid"))


def plot_gain_bar(results, save_path="gain_bar.png"):
    """
    Bar plot of mean gain (+/- SE) per method, with worst gain overlaid
    as a point marker on the same axes. A dashed horizontal line at 0 marks
    the random-baseline reference.
    """
    methods = list(results.keys())
    means = [results[m]['mean_gain'] for m in methods]
    ses = [results[m]['mean_gain_se'] for m in methods]
    worst = [results[m]['worst_gain'] for m in methods]

    x = np.arange(len(methods))
    colors = [METHOD_COLOR_LIST.get(m, "gray") for m in methods]

    with mplscience.style_context():
        fig, ax = plt.subplots(figsize=(4, 3.3))

        bars = ax.bar(x, means, yerr=ses, capsize=4, color=colors)
        # ax.scatter(x, worst, marker="D", color="firebrick", zorder=3)
        ax.axhline(0, linestyle="--", color="gray", linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_ylabel("Relative Gain over Random", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)

        plt.tight_layout()
        outpath_png = save_path
        outpath_pdf = os.path.splitext(save_path)[0] + ".pdf"
        plt.savefig(outpath_png)
        plt.savefig(outpath_pdf, dpi=300, bbox_inches='tight')
        plt.close(fig)


def write_gain_csv(mean_gain_by_dataset, mean_rank_by_dataset, results, dataset_order, save_path="gain_summary.csv"):
    """
    Wide-format CSV, one row per method: per-dataset mean gain (gain) and
    rank, plus the overall worst gain/rank across datasets (from `results`).
    """
    header = (
        ["Method"]
        + [f"{d}_gain" for d in dataset_order]
        + ["Worst_Gain"]
        + [f"{d}_rank" for d in dataset_order]
        + ["Worst_Rank"]
    )
    rows = []
    for method, stats in results.items():
        gain_by_dataset = mean_gain_by_dataset.get(method, {})
        rank_by_dataset = mean_rank_by_dataset.get(method, {})
        row = [method]
        row += [f"{gain_by_dataset[d]:.4f}" if d in gain_by_dataset else "" for d in dataset_order]
        row.append(f"{stats['worst_gain']:.4f}")
        row += [f"{int(rank_by_dataset[d])}" if d in rank_by_dataset else "" for d in dataset_order]
        row.append(f"{int(stats['worst_rank'])}")
        rows.append(row)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def print_per_dataset_perf_table(mean_perf_by_dataset, dataset_order):
    """
    Raw mean test_perf at each dataset's max_rounds (averaged over splits
    and runs), one row per method including Random, one column per dataset
    — the raw numbers mean_gain_d/mean_rank_d are derived from.
    """
    print("Per-dataset mean performance at max_rounds:")
    rows = []
    for method, by_dataset in mean_perf_by_dataset.items():
        rows.append([method] + [
            f"{by_dataset[d]:.4f}" if d in by_dataset else "-"
            for d in dataset_order
        ])
    print(tabulate.tabulate(rows, headers=["Strategy"] + dataset_order, tablefmt="fancy_grid"))


def print_per_dataset_table(mean_gain_by_dataset, mean_rank_by_dataset, dataset_order):
    """
    Per-dataset breakdown (gain and rank), printed before the aggregate
    table so a strong win on some datasets and a wash on others isn't
    hidden by the equal-weighted cross-dataset average.
    """
    print("Per-dataset mean gain:")
    rows = []
    for method, by_dataset in mean_gain_by_dataset.items():
        rows.append([method] + [
            f"{by_dataset[d]:.3f}" if d in by_dataset else "-"
            for d in dataset_order
        ])
    print(tabulate.tabulate(rows, headers=["Strategy"] + dataset_order, tablefmt="fancy_grid"))

    print("Per-dataset mean rank:")
    rows = []
    for method, by_dataset in mean_rank_by_dataset.items():
        rows.append([method] + [
            f"{by_dataset[d]:.2f}" if d in by_dataset else "-"
            for d in dataset_order
        ])
    print(tabulate.tabulate(rows, headers=["Strategy"] + dataset_order, tablefmt="fancy_grid"))


def parse_strat_list_from_dirname(name):
    if name.startswith("["):
        try:
            import ast

            end = name.find("]")
            substr = name[: end + 1]
            lst = ast.literal_eval(substr)
            return [s.strip() for s in lst]
        except Exception:
            return []
    return []


def load_perf_array(res_json, perf_field, metric_key, higher_is_better=False):
    """
    Load performance numbers from a performance_records.json file.

    Returns: numpy array shaped (n_splits, n_runs, n_rounds)
    """
    with open(res_json) as f:
        res = json.load(f)
    dsplit_keys = sorted([k for k in res.keys() if k.startswith('dsplit_')])
    vals = []
    for dkey in dsplit_keys:
        split = res[dkey]
        run_keys = sorted([k for k in split.keys() if k.startswith('run_')])
        run_vals = []
        for rkey in run_keys:
            rd_list = split[rkey]
            try:
                rd_list = sorted(rd_list, key=lambda d: d.get('round', 0))
            except Exception:
                pass
            per_round = []
            for rd in rd_list:
                if perf_field in rd and rd[perf_field] is not None:
                    perf = rd[perf_field]
                    if metric_key is None:
                        if isinstance(perf, (int, float)):
                            per_round.append(float(perf))
                        else:
                            per_round.append(np.nan)
                    else:
                        if metric_key in perf:
                            per_round.append(float(perf[metric_key]))
                        elif metric_key == 'mse' and 'loss' in perf:
                            # Some methods (e.g. AutoAL) log regression error
                            # under 'loss' instead of 'mse' - same quantity.
                            per_round.append(float(perf['loss']))
                        else:
                            per_round.append(np.nan)
            run_vals.append(per_round)
        vals.append(run_vals)
    arr = np.array(vals, dtype=object)
    n_splits = arr.shape[0]
    n_runs = max(len(arr[i]) for i in range(n_splits))
    n_rounds = max(len(arr[i][j]) for i in range(n_splits) for j in range(len(arr[i])))
    out = np.full((n_splits, n_runs, n_rounds), np.nan, dtype=float)
    for i in range(n_splits):
        for j in range(len(arr[i])):
            rvals = arr[i][j]
            out[i, j, :len(rvals)] = rvals
    return out


def get_round_values(arr, max_rounds):
    """
    arr: (n_splits, n_runs, n_rounds)
    Returns: (n_splits, n_runs) at round index `max_rounds - 1` (clamped to
    the array's actual last round if it has fewer rounds than max_rounds).
    """
    n_rounds = arr.shape[2]
    round_idx = min(max_rounds, n_rounds) - 1
    return arr[:, :, round_idx]


def compute_gain(method_final, p_rand_final, best_pure_final):
    """
    (method - p_rand) / (p_best - p_rand). Called on per-dataset scalars
    (mean performance over all (split, run) cells) rather than per-cell
    arrays, to avoid small-denominator instability (a mean-of-ratios can be
    dominated by outlier cells where p_best - p_rand is near zero; a
    ratio-of-means is not). Undefined (denom == 0) returns NaN.
    """
    denom = best_pure_final - p_rand_final
    if denom == 0:
        return float('nan')
    return (method_final - p_rand_final) / denom


def rank_methods_by_mean_perf(mean_perfs, higher_is_better):
    """
    mean_perfs: (n_methods,) dataset-level mean performance per method
    (the same mean used for gain). Returns integer ranks of the same
    shape; rank 1 = best. NaNs stay NaN and are excluded from ranking.
    """
    ranks = np.full_like(mean_perfs, np.nan)
    valid = ~np.isnan(mean_perfs)
    if not valid.any():
        return ranks
    order = np.argsort(-mean_perfs[valid] if higher_is_better else mean_perfs[valid])
    valid_ranks = np.empty(valid.sum())
    valid_ranks[order] = np.arange(1, valid.sum() + 1)
    ranks[valid] = valid_ranks
    # breakpoint()
    return ranks


def main():
    dataset_info = {
        "cf10_3comp": {
            "res_fold": "cifar10",
            "metric_key": "acc",
            "higher_is_better": True,
            "max_rounds": 10,
        },
        "cf10_8comp": {
            "res_fold": "cifar10",
            "metric_key": "acc",
            "higher_is_better": True,
            "max_rounds": 10,
        },
        "kegg_4comp": {
            "res_fold": "kegg_undir_uci",
            "metric_key": "mse",
            "higher_is_better": False,
            "max_rounds": 10,
        },
        "sarcos_4comp": {
            "res_fold": "sarcos",
            "metric_key": "mse",
            "higher_is_better": False,
            "max_rounds": 10,
        },
        "diamonds_4comp": {
            "res_fold": "diamonds",
            "metric_key": "mse",
            "higher_is_better": False,
            "max_rounds": 10,
        },
        "bio_macrophage": {
            "res_fold": "bio_macrophage",
            "metric_key": "auprc",
            "higher_is_better": True,
            "max_rounds": 6,
        },
        "bio_tcell": {
            "res_fold": "bio_tcell",
            "metric_key": "auprc",
            "higher_is_better": True,
            "max_rounds": 6,
        },
    }

    pure_strat_folders_dict = {
        "cf10_3comp": [
            ## Add the folder paths of pure strategy performances in alphabetical order. First one should always be random.
        ],
        "cf10_8comp": [
            ## Add the folder paths of pure strategy performances in alphabetical order. First one should always be random.
        ],
        "kegg_4comp": [
            ## Add the folder paths of pure strategy performances in alphabetical order. First one should always be random.
        ],
        "sarcos_4comp": [
            ## Add the folder paths of pure strategy performances in alphabetical order. First one should always be random.
        ],
        "diamonds_4comp": [
            ## Add the folder paths of pure strategy performances in alphabetical order. First one should always be random.
        ],
        "bio_macrophage": [
            ## Add the folder paths of pure strategy performances in alphabetical order. First one should always be random.
        ],
        "bio_tcell": [
            ## Add the folder paths of pure strategy performances in alphabetical order. First one should always be random.
        ],
    }
    comp_meth_folders_dict = {
        "cf10_3comp": [
            # Add the folder names of adaptive methods and random in the order described in label_list. Random should be first.
        ],
        "cf10_8comp": [
            # Add the folder names of adaptive methods and random in the order described in label_list. Random should be first.
        ],
        "kegg_4comp": [
            # Add the folder names of adaptive methods and random in the order described in label_list. Random should be first.
        ],
        "sarcos_4comp": [
            # Add the folder names of adaptive methods and random in the order described in label_list. Random should be first.
        ],
        "diamonds_4comp": [
            # Add the folder names of adaptive methods and random in the order described in label_list. Random should be first.
        ],
        "bio_macrophage": [
            # Add the folder names of adaptive methods and random in the order described in label_list. Random should be first.
        ],
        "bio_tcell": [
            # Add the folder names of adaptive methods and random in the order described in label_list. Random should be first.
        ],
    }
    label_list = ["Random", "ALBL", "UCB", "SCRiBLe", "SelectAL", "AutoAL", "Equal Split", "FractAL"]
    perf_field = 'test_perf'

    reportable_labels = label_list[1:]  # exclude "Random"
    mean_gain_by_dataset = {label: {} for label in reportable_labels}
    mean_rank_by_dataset = {label: {} for label in reportable_labels}
    mean_perf_by_dataset = {label: {} for label in label_list}

    for data in list(dataset_info.keys()):
        print(data)
        metric = dataset_info[data]['metric_key']
        higher_is_better = dataset_info[data]['higher_is_better']
        max_rounds = dataset_info[data]['max_rounds']
        pure_strat_folders = pure_strat_folders_dict[data]
        comp_meth_folders = comp_meth_folders_dict[data]
        results_dir = os.path.join("./results", dataset_info[data]['res_fold'])

        # --- Best pure-strategy final-round performance per (split, run) ---
        pure_finals = []
        for d in pure_strat_folders:
            folder = os.path.join(results_dir, d)
            if os.path.isdir(folder):
                res_json = os.path.join(folder, 'performance_records.json')
            else:
                raise FileNotFoundError(f'{folder} not found or is not a directory')
            if not os.path.exists(res_json):
                raise FileNotFoundError(f'{res_json} not found')
            arr = load_perf_array(res_json, perf_field, metric, higher_is_better)
            pure_finals.append(get_round_values(arr, max_rounds))

        shapes = [a.shape for a in pure_finals]
        n_splits, n_runs = shapes[0]
        for s in shapes:
            if s != (n_splits, n_runs):
                raise ValueError('Pure strategy result shapes do not match across inputs')

        pure_finals_stack = np.stack(pure_finals, axis=0)  # (n_pure, n_splits, n_runs)
        mean_pure = np.nanmean(pure_finals_stack, axis=(1,2))
        if higher_is_better:
            best_pure_final = np.nanmax(mean_pure, axis=0)
        else:
            best_pure_final = np.nanmin(mean_pure, axis=0)

        def load_method_final(mdir):
            folder = os.path.join(results_dir, mdir)
            if os.path.isdir(folder):
                res_json = os.path.join(folder, 'performance_records.json')
            else:
                raise FileNotFoundError(f'{folder} not found or is not a directory')
            if not os.path.exists(res_json):
                return None
            arr = load_perf_array(res_json, perf_field, metric, higher_is_better)
            method_final = get_round_values(arr, max_rounds)
            if method_final.shape != (n_splits, n_runs):
                print(f'Warning: method result shape {method_final.shape} does not match pure-data shape {(n_splits, n_runs)}; attempting to align')
                aligned = np.full((n_splits, n_runs), np.nan)
                s_lim = min(n_splits, method_final.shape[0])
                r_lim = min(n_runs, method_final.shape[1])
                aligned[:s_lim, :r_lim] = method_final[:s_lim, :r_lim]
                method_final = aligned
            return method_final

        # --- Random baseline (comp_meth_folders[0]) ---
        p_rand_final = load_method_final(comp_meth_folders[0])
        if p_rand_final is None:
            print(f'Warning: Random baseline not found for {data} — skipping dataset')
            continue

        # --- Raw performance + gain per adaptive method (excludes Random) ---
        method_finals = []
        method_labels_present = []
        for idx in range(1, len(comp_meth_folders)):
            mdir = comp_meth_folders[idx]
            method_final = load_method_final(mdir)
            if method_final is None:
                print(f'Warning: {mdir} not found — skipping')
                continue
            method_finals.append(method_final)
            method_labels_present.append(label_list[idx])

        if not method_finals:
            continue

        perf_stack = np.stack(method_finals, axis=0)  # (n_methods, n_splits, n_runs)
        mean_perfs = np.array([np.nanmean(perf_stack[m_idx]) for m_idx in range(len(method_labels_present))])
        dataset_ranks = rank_methods_by_mean_perf(mean_perfs, higher_is_better)

        mean_p_rand = np.nanmean(p_rand_final)
        mean_p_best = best_pure_final
        mean_perf_by_dataset['Random'][data] = float(mean_p_rand)
        for m_idx, label in enumerate(method_labels_present):
            mean_p_i = mean_perfs[m_idx]
            mean_gain_d = compute_gain(mean_p_i, mean_p_rand, mean_p_best)
            mean_gain_by_dataset[label][data] = float(mean_gain_d)
            mean_rank_by_dataset[label][data] = float(dataset_ranks[m_idx])
            mean_perf_by_dataset[label][data] = float(mean_p_i)

    print_per_dataset_perf_table(mean_perf_by_dataset, list(dataset_info.keys()))
    print_per_dataset_table(mean_gain_by_dataset, mean_rank_by_dataset, list(dataset_info.keys()))

    results = {}
    for label in reportable_labels:
        gain_vals = np.array(list(mean_gain_by_dataset[label].values()), dtype=float)
        rank_vals = np.array(list(mean_rank_by_dataset[label].values()), dtype=float)
        n_gain = np.sum(~np.isnan(gain_vals))
        n_rank = np.sum(~np.isnan(rank_vals))
        if n_gain == 0 and n_rank == 0:
            continue
        results[label] = {
            'mean_gain': float(np.nanmean(gain_vals)) if n_gain else float('nan'),
            'mean_gain_se': float(np.nanstd(gain_vals)) / np.sqrt(n_gain) if n_gain else float('nan'),
            'median_gain': float(np.nanmedian(gain_vals)) if n_gain else float('nan'),
            'worst_gain': float(np.nanmin(gain_vals)) if n_gain else float('nan'),
            'mean_rank': float(np.nanmean(rank_vals)) if n_rank else float('nan'),
            'mean_rank_se': float(np.nanstd(rank_vals)) / np.sqrt(n_rank) if n_rank else float('nan'),
            'median_rank': float(np.nanmedian(rank_vals)) if n_rank else float('nan'),
            'worst_rank': float(np.nanmax(rank_vals)) if n_rank else float('nan'),
        }

    print_table(results)
    plot_gain_bar(results)
    write_gain_csv(mean_gain_by_dataset, mean_rank_by_dataset, results, list(dataset_info.keys()))


if __name__ == '__main__':
    main()
