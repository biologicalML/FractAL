"""
Compute precisison@k of methods against pure-strategy ground truth.
"""
import json
import os
import numpy as np
import tabulate
from scipy import stats
from significance_test import one_sample_split_test, paired_split_test
import argparse

def print_table(results, label_list):
    rows = []
    data_list = []
    for method in results.keys():
        row = [method]
        for data in results[method].keys():
            data_list.append(data)
            row.append(f"{results[method][data]['mean_prec']:.2f} ± {results[method][data]['prec_se']:.2f}")
        rows.append(row)
    headers = ["Strategy"] + data_list
    print(tabulate.tabulate(rows, headers=headers, tablefmt="fancy_grid"))

def print_pval_table(pvals, method_label):
    rows = [[dataset, f"{p:.4g}"] for dataset, p in pvals.items()]
    print(tabulate.tabulate(rows, headers=["Dataset", f"p-value ({method_label})"], tablefmt="fancy_grid"))

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
    # collect splits sorted by key
    dsplit_keys = sorted([k for k in res.keys() if k.startswith('dsplit_')])
    vals = []
    for dkey in dsplit_keys:
        split = res[dkey]
        run_keys = sorted([k for k in split.keys() if k.startswith('run_')])
        run_vals = []
        for rkey in run_keys:
            rd_list = split[rkey]
            # sort by round if available
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
                        else:
                            per_round.append(np.nan)
            run_vals.append(per_round)
        vals.append(run_vals)
    arr = np.array(vals, dtype=object)
    # convert to rectangular numeric array, fill missing with nan
    n_splits = arr.shape[0]
    n_runs = max(len(arr[i]) for i in range(n_splits))
    n_rounds = max(len(arr[i][j]) for i in range(n_splits) for j in range(len(arr[i])))
    out = np.full((n_splits, n_runs, n_rounds), np.nan, dtype=float)
    for i in range(n_splits):
        for j in range(len(arr[i])):
            rvals = arr[i][j]
            out[i, j, :len(rvals)] = rvals
    return out

def load_fraction_array(res_json, strategy_names, ordered_strats, label):
    """
    Load Fraction data from a performance_records.json file.

    Returns: numpy array shaped (n_splits, n_runs, n_rounds, n_strats)
    where order of strategies follows `ordered_strats`.
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
            per_round = []
            for rd in rd_list:
                if label in ["SelectAL", "FractAL-CB"]:
                    frac = rd['influences']
                else:
                    frac = rd['fractions']
                if frac is None:
                    per_round.append([np.nan] * len(strategy_names))
                    continue
                # assume same order as strategy_names
                row = []
                for strat in ordered_strats:
                    s_idx = strategy_names.index(strat)
                    row.append(float(frac[s_idx]))
                per_round.append(row)
            run_vals.append(per_round)
        vals.append(run_vals)
    arr = np.array(vals, dtype=object)
    n_splits = arr.shape[0]
    n_runs = max(len(arr[i]) for i in range(n_splits))
    n_rounds = max(len(arr[i][j]) for i in range(n_splits) for j in range(len(arr[i])))
    n_strats = len(strategy_names)
    out = np.full((n_splits, n_runs, n_rounds, n_strats), np.nan, dtype=float)
    for i in range(n_splits):
        for j in range(len(arr[i])):
            for r_idx, row in enumerate(arr[i][j]):
                out[i, j, r_idx, : len(row)] = row
    return out

def compute_round_wise_top_k(y_true, y_probs, max_rounds, higher_is_better, k=5):
    """
    y_true: (n_samples, n_rounds, n_strats)
    y_probs: (n_samples, n_rounds, n_strats)
    """
    
    y_true = y_true[:, 2:max_rounds, :]
    y_probs = y_probs[:, 1:max_rounds-1, :] #The fractions for current round are recorded in the dict of previous round
    n_samples, n_rounds, n_strats = y_true.shape
    # 1. Get indices of top K predictions across the class axis (axis=2)
    if higher_is_better:
        # argsort ascending, take last k (largest)
        perf_topk = np.argsort(y_true, axis=-1)[..., -k:]
    else:
        # argsort ascending, take first k (smallest)
        perf_topk = np.argsort(y_true, axis=-1)[..., :k]

    # --- Top-K indices for fractions (always higher is better) ---
    frac_topk = np.argsort(y_probs, axis=-1)[..., -k:]

    # --- Compute overlap count per (split, seed, round) cell ---
    overlap_counts = np.zeros((n_samples, n_rounds), dtype=int)

    for s in range(n_samples):
        for se in range(n_rounds):
            perf_set = set(perf_topk[s, se])
            frac_set = set(frac_topk[s, se])
            overlap_counts[s, se] = len(perf_set & frac_set)

    overlap_fractions = overlap_counts / k
    round_precisions = np.nanmean(overlap_fractions, axis=0)
    
    return round_precisions, overlap_fractions

def count_overlaps_with_best(means, stds, higher_is_better):
    # 1. Find the index of the highest mean
    if higher_is_better:
        best_idx = np.argmax(means)
    else:
        best_idx = np.argmin(means)
    
    # 2. Define the interval for the best strategy
    best_mean = means[best_idx]
    best_std = stds[best_idx]
    best_low = best_mean - best_std
    best_high = best_mean + best_std
    
    overlapping_indices = []
    
    # 3. Check every other strategy against the best
    for i in range(len(means)):
        if i == best_idx:
            continue  # Don't count the best one against itself
            
        current_low = means[i] - stds[i]
        current_high = means[i] + stds[i]
        
        # Overlap logic: 
        # The ranges overlap if the current's high is above the best's low 
        # AND the current's low is below the best's high.
        if current_high >= best_low and current_low <= best_high:
            overlapping_indices.append(i)
            
    return best_idx, overlapping_indices

def get_k_for_dataset(true, max_rounds, higher_is_better):
    # true is (n_splits, n_runs, n_rounds, n_strats)
    true = true[:,:,2:max_rounds,:] ## The performance is always recorded for the current round
    n_rounds = true.shape[2]
    true_avg = np.nanmean(true, axis=0)
    true_avg_std = np.nanstd(true_avg, axis=0) #n_rounds, n_strats
    true_avg_mean = np.nanmean(true_avg, axis=0)
    rd_k_vals = []
    for rd in range(n_rounds):
        _,overlaps = count_overlaps_with_best(true_avg_mean[rd], true_avg_std[rd], higher_is_better)
        rd_k_vals.append(1 + len(overlaps))
    k = stats.mode(rd_k_vals)[0]
    return k

def main(args):
    dataset_info = {
        "cf10_3comp":{
            "res_fold": "cifar10",
            "metric_key": "acc",
            "higher_is_better": True,
            "max_rounds": 10,
        },
        "cf10_8comp":{
            "res_fold": "cifar10",
            "metric_key": "acc",
            "higher_is_better": True,
            "max_rounds": 10,
        },
        "kegg_4comp":{
            "res_fold": "kegg_undir_uci",
            "metric_key": "mse",
            "higher_is_better": False,
            "max_rounds": 10,
        },
        "sarcos_4comp":{
            "res_fold": "sarcos",
            "metric_key": "mse",
            "higher_is_better": False,
            "max_rounds": 10,
        },
        "diamonds_4comp":{
            "res_fold": "diamonds",
            "metric_key": "mse",
            "higher_is_better": False,
            "max_rounds": 10,
        },
        "bio_macrophage":{
            "res_fold": "bio_macrophage",
            "metric_key": "auprc",
            "higher_is_better": True,
            "max_rounds": 6,
        },
        "bio_tcell":{
            "res_fold": "bio_tcell",
            "metric_key": "auprc",
            "higher_is_better": True,
            "max_rounds": 6,
        }
    }

    pure_strat_folders_dict = {
        "cf10_3comp": [
        ########CIFAR 10 3 STRAT
        #<Add folder names from the results directory in alphabetical order of pure strategies.. Random first and then in alphabetical order>
        ],
        ####-----------------------------------------------------------------------------------------------------------------------------------------------------------------
        ########CIFAR 10 8 STRAT
        "cf10_8comp": [
        #<Add folder names from the results directory in alphabetical order of pure strategies.. Random first and then in alphabetical order>
        ],
        ####-----------------------------------------------------------------------------------------------------------------------------------------------------------------
        ########Tabular Regression
        "kegg_4comp": [
        #<Add folder names from the results directory in alphabetical order of pure strategies.. Random first and then in alphabetical order>
        ],
        "sarcos_4comp": [
        #<Add folder names from the results directory in alphabetical order of pure strategies.. Random first and then in alphabetical order>
        ],
        "diamonds_4comp": [
        #<Add folder names from the results directory in alphabetical order of pure strategies.. Random first and then in alphabetical order>
        ],
        ####-----------------------------------------------------------------------------------------------------------------------------------------------------------------
        ########TCell
        "bio_macrophage": [
        #<Add folder names from the results directory in alphabetical order of pure strategies.. Random first and then in alphabetical order>
        ],
        "bio_tcell": [
        #<Add folder names from the results directory in alphabetical order of pure strategies.. Random first and then in alphabetical order>
        ],
    }
    comp_meth_folders_dict = {
        "cf10_3comp": [
        ########CIFAR 10 3 STRAT
        #<Add folder names from the results directory in order of the labels..
        #labels = ["SCRiBLe","SelectAL", "FractAL"]
        ],
        ####-----------------------------------------------------------------------------------------------------------------------------------------------------------------
        "cf10_8comp": [
        #<Add folder names from the results directory in order of the labels..
        ],
        ####-----------------------------------------------------------------------------------------------------------------------------------------------------------------
        "kegg_4comp": [
        #<Add folder names from the results directory in order of the labels..
        ],
        "sarcos_4comp": [
        #<Add folder names from the results directory in order of the labels..
        ],
        "diamonds_4comp": [
        #<Add folder names from the results directory in order of the labels..
        ],
        ####-----------------------------------------------------------------------------------------------------------------------------------------------------------------
        "bio_macrophage": [
        #<Add folder names from the results directory in order of the labels..
        ],
        "bio_tcell": [
        #<Add folder names from the results directory in order of the labels..
        ],

    }
    label_list = ["SCRiBLe", "SelectAL", "FractAL"]
    perf_field = 'test_perf'
        
    results ={}
    for label in label_list:
        results[label] = {}
    fractal_pvals = {}
    fractal_vs_selectal_pvals = {}
    for data in list(dataset_info.keys()):
        print(data)
        max_rounds = dataset_info[data]['max_rounds']
        metric = dataset_info[data]['metric_key']
        higher_is_better = dataset_info[data]['higher_is_better']
        pure_strat_folders = pure_strat_folders_dict[data]
        comp_meth_folders = comp_meth_folders_dict[data]
        strategy_names = parse_strat_list_from_dirname(comp_meth_folders[0])
        ordered_strats = []
        if 'random' in strategy_names:
            ordered_strats.append('random')
        ordered_strats += sorted([s for s in strategy_names if s != 'random'])
        results_dir = os.path.join("./results", dataset_info[data]['res_fold'])
        # load all pure arrays
        pure_arrays = []
        for d in pure_strat_folders:
            path = d
            folder = os.path.join(results_dir, d)
            if os.path.isdir(folder):
                res_json = os.path.join(folder, 'performance_records.json')
            else:
                raise FileNotFoundError(f'{folder} not found or is not a directory')
            if not os.path.exists(res_json):
                raise FileNotFoundError(f'{res_json} not found')
            arr = load_perf_array(res_json, perf_field, metric, higher_is_better)
            pure_arrays.append(arr)

        # verify shapes align
        shapes = [a.shape for a in pure_arrays]
        n_splits = shapes[0][0]
        n_runs = shapes[0][1]
        n_rounds = shapes[0][2]
        for s in shapes:
            if s[0] != n_splits or s[1] != n_runs or s[2] != n_rounds:
                raise ValueError('Pure strategy result shapes do not match across inputs')

        n_strats = len(pure_arrays)
        # stack to (n_strats, n_splits, n_runs, n_rounds)
        stacked = np.stack(pure_arrays, axis=0)
        # reorder to (n_splits, n_runs, n_rounds, n_strats)
        stacked = np.transpose(stacked, (1, 2, 3, 0))
        if args.worst:
            stacked = stacked *-1
        k_val = get_k_for_dataset(stacked, max_rounds, higher_is_better)

        ## To correct for trivial k-values
        if args.worst:
            if data == "diamonds_4comp":
                k_val = 2
        else:
            if data == "diamonds_4comp" or data == "cf10_8comp" or data == "sarcos_4comp":
                k_val = 2
        
        data = data + f" (k={k_val})"
        true_vals = np.reshape(stacked, (-1, n_rounds, n_strats))

        precision_by_method = {}
        for idx, mdir in enumerate(comp_meth_folders):
            path = mdir
            folder = os.path.join(results_dir, mdir)
            if os.path.isdir(folder):
                res_json = os.path.join(folder, 'performance_records.json')
            else:
                raise FileNotFoundError(f'{folder} not found or is not a directory')
            if not os.path.exists(res_json):
                print(f'Warning: {res_json} not found — skipping')
                continue
            frac_arr = load_fraction_array(res_json, strategy_names, ordered_strats, label_list[idx])
            # check shapes
            if frac_arr.shape[0] != n_splits or frac_arr.shape[1] != n_runs or frac_arr.shape[2] < n_rounds:
                print(f'Warning: fraction array shape {frac_arr.shape} does not match pure-data shape {(n_splits,n_runs,n_rounds)}; attempting to align')
            # Trim/pad frac_arr to match n_rounds
            frac_arr = frac_arr[:, :n_runs, :n_rounds, :n_strats]
            frac_arr = np.reshape(frac_arr, (-1, n_rounds, n_strats))
            if args.worst:
                frac_arr = frac_arr * -1
            # compute accuracy per round averaged over splits and runs
            prec_per_round, overlap_fractions = compute_round_wise_top_k(true_vals, frac_arr, max_rounds, higher_is_better, k=k_val)
            mean_prec = float(np.nanmean(prec_per_round))
            prec_se = float(np.nanstd(prec_per_round))/ np.sqrt(prec_per_round.shape[0])
            
            if label_list[idx] in ("FractAL", "SelectAL"):
                n_test_rounds = overlap_fractions.shape[1]         # note: this is max_rounds - 3 due to the slicing above
                overlap_by_split_run = overlap_fractions.reshape(n_splits, n_runs, n_test_rounds)
                per_split_seed_mean = np.nanmean(overlap_by_split_run, axis=2)
                precision_by_method[label_list[idx]] = per_split_seed_mean

                if label_list[idx] == "FractAL":
                    chance = k_val/n_strats
                    test_result = one_sample_split_test(per_split_seed_mean, chance, higher_is_better=True, test="ttest", alternative="greater")
                    fractal_pvals[data] = test_result['p_value']
            results[label_list[idx]][data] = {
                'mean_prec': mean_prec,
                'prec_se': prec_se,
            }

        if "FractAL" in precision_by_method and "SelectAL" in precision_by_method:
            pair_result = paired_split_test(precision_by_method["FractAL"], precision_by_method["SelectAL"], higher_is_better=True, test="ttest")
            # # breakpoint()
            # if data.startswith("bio") or data.startswith("kegg_4comp"):
            #     # print(pair_result)
            #     pair_result = paired_split_test(precision_by_method["FractAL"], precision_by_method["SelectAL"], higher_is_better=True, test="wilcoxon")
            fractal_vs_selectal_pvals[data] = pair_result['p_value']

    print_table(results, label_list)
    print_pval_table(fractal_pvals, "FractAL vs chance")
    print_pval_table(fractal_vs_selectal_pvals, "FractAL vs SelectAL")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--worst",
            action="store_true",
            help="Whether to use the worst-case scenario for the allocation diagnostics",
        )
    args = parser.parse_args()
    main(args)
