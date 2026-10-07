import os
import json
import numpy as np
import matplotlib.pyplot as plt
import mplscience
from args import read_args

def iter_runs_container(container):
    if isinstance(container, dict):
        run_keys = [k for k in container.keys() if str(k).startswith("run_")]
        if run_keys:
            run_keys = sorted(run_keys, key=lambda x: int(x.split("_")[-1]))
            for k in run_keys:
                yield container[k]
            return
        for k in sorted(container.keys()):
            yield container[k]
        return
    if isinstance(container, list):
        for item in container:
            yield item


def find_performance_json(dirpath):
    p = os.path.join(dirpath, "performance_records.json")
    if os.path.exists(p):
        return p
    for root, dirs, files in os.walk(dirpath):
        if "performance_records.json" in files:
            return os.path.join(root, "performance_records.json")
    return None


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

label_dict = {
    "random": "Random",
    "coreset": "CoreSet",
    "probcover": "ProbCover",
    "typiclust": "TypiClust",
    "badge": "BADGE",
    "maxent": "MaxEnt",
    "minmargin": "MinMargin",
    "leastconf": "LeastConf",
}
color_list = {
    "Random": "black",
    "BADGE": "yellow",
    "CoreSet": "blue",
    "LeastConf": "brown",
    "MaxEnt": "orange",
    "MinMargin": "purple",
    "ProbCover": "red",
    "TypiClust": "green",
}
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["font.family"] = "DejaVu Sans"

LABEL_SIZE = 18
TITLE_SIZE = 15
TICK_SIZE = 18
LEGEND_SIZE = 12

def plot_strat_evolution_for_folder(folder, results_dir, label, plot_dir, max_rounds, legend_only, legend_outside):
    """
    Build and save one strategy-proportion evolution figure for a single
    method folder (dataset's fraction allocations across strategies, per
    AL round).
    """
    folder_path = os.path.join(results_dir, folder)
    perf_json = find_performance_json(folder_path)
    if perf_json is None:
        print(f"performance_records.json not found for {folder}")
        return

    with open(perf_json) as f:
        res = json.load(f)
    strats_in_comp = parse_strat_list_from_dirname(folder)
    if not strats_in_comp:
        return

    data_block = {s: {} for s in strats_in_comp}
    for dsplit_key in sorted([k for k in res.keys() if str(k).startswith("dsplit_")]):
        container = res[dsplit_key]
        for run_list in iter_runs_container(container):
            rd_counter = 0
            for rd in run_list:
                if "round" not in rd:
                    continue
                rnum = rd_counter
                fracs = rd.get("fractions", [])
                for idx, strat in enumerate(strats_in_comp):
                    val = float(fracs[idx]) if idx < len(fracs) else 0.0
                    data_block[strat].setdefault(rnum, {}).setdefault(dsplit_key, []).append(val)
                rd_counter += 1

    ordered_strats = []
    if 'random' in strats_in_comp:
        ordered_strats.append('random')
    ordered_strats += sorted([s for s in strats_in_comp if s != 'random'])

    fig, ax = plt.subplots(figsize=(4, 3.3))
    plotted_any = False

    with mplscience.style_context():
        for strat in ordered_strats:
            data = data_block.get(strat, {})
            rounds = sorted([r for r in data.keys() if r is not None and r >= 0])
            plot_rounds = [r for r in rounds if r > 0] or rounds
            if max_rounds is not None:
                plot_rounds = [r for r in plot_rounds if (r + 1) <= max_rounds]

            x_vals = np.array([r + 1 for r in plot_rounds[:-1]])
            y_means = []
            y_stds = []
            for r in plot_rounds[:-1]:
                round_dict = data.get(r, {})
                split_means = [np.mean(vals) if len(vals) > 0 else 0.0 for vals in round_dict.values()]
                if len(split_means) == 0:
                    y_means.append(0.0)
                    y_stds.append(0.0)
                else:
                    y_means.append(float(np.mean(split_means)))
                    y_stds.append(float(np.std(split_means)))

            if len(x_vals) == 0 or len(y_means) == 0:
                continue

            color = color_list[label_dict[strat]] if label_dict[strat] in color_list else None
            line, = ax.plot(x_vals, y_means, label=label_dict[strat], color=color, lw=2.5, marker='o', markersize=4)
            ax.fill_between(x_vals, np.array(y_means) - np.array(y_stds), np.array(y_means) + np.array(y_stds),
                            color=line.get_color(), alpha=0.15)
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            return

        ax.set_xticks(x_vals)
        ax.set_xlabel('AL Iteration', fontsize=LABEL_SIZE)
        ax.set_ylabel('Average Proportion', fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE)

        handles, labels = ax.get_legend_handles_labels()
        os.makedirs(plot_dir, exist_ok=True)
        dataset_name = os.path.basename(plot_dir)
        outpath = os.path.join(plot_dir, f'{dataset_name}_{len(ordered_strats)}comp_{label}_strat_evolution.pdf')
        outpath_png = os.path.join(plot_dir, f'{dataset_name}_{len(ordered_strats)}comp_{label}_strat_evolution.png')

        if legend_only:
            # Save main plot without legend
            plt.tight_layout()
            plt.savefig(outpath_png)
            plt.savefig(outpath, dpi=300, bbox_inches='tight')

        else:
            if legend_outside:
                # Place legend to the right outside the axes
                if len(labels) > 0:
                    ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=LEGEND_SIZE)
                # leave space on the right when saving
                plt.tight_layout(rect=[0, 0, 0.85, 1])
            else:
                # Default: inside the plot (upper right)
                if len(labels) > 0:
                    ax.legend(loc='upper right', frameon=True, fontsize=LEGEND_SIZE)
                plt.tight_layout()

            plt.savefig(outpath_png)
            plt.savefig(outpath, dpi=300, bbox_inches='tight')

        plt.close(fig)
        print(f"Saved: {outpath}")


def plot_neurips_strat(comp_meth_folders_dict, dataset_info, label_list=None, legend_only=False, legend_outside=False):
    """
    For each dataset in comp_meth_folders_dict, save the strategy-wise
    proportion evolution plot for every non-skipped method folder.
    """
    if label_list is None:
        label_list = ["Random", "ALBL", "UCB", "SCRiBLe", "SelectAL", "AutoAL", "FractAL"]
    skip_labels = {"Random", "AutoAL"}

    for data in comp_meth_folders_dict:
        info = dataset_info[data]
        results_dir = os.path.join("./results", info["res_fold"])
        max_rounds = info["max_rounds"]
        plot_dir = os.path.join("plots", data)

        for idx, folder in enumerate(comp_meth_folders_dict[data]):
            if idx < len(label_list):
                label = label_list[idx]
            else:
                print(f"Warning: no label for index {idx} in {data}; using folder name")
                label = folder
            if label in skip_labels:
                continue
            plot_strat_evolution_for_folder(folder, results_dir, label, plot_dir, max_rounds, legend_only, legend_outside)


def main():
    args = read_args()

    dataset_info = {
        "cf10_3comp": {"res_fold": "cifar10", "max_rounds": 10},
        "cf10_8comp": {"res_fold": "cifar10", "max_rounds": 10},
        "kegg_4comp": {"res_fold": "kegg_undir_uci", "max_rounds": 10},
        "sarcos_4comp": {"res_fold": "sarcos", "max_rounds": 10},
        "diamonds_4comp": {"res_fold": "diamonds", "max_rounds": 10},
        "bio_macrophage": {"res_fold": "bio_macrophage", "max_rounds": 6},
        "bio_tcell": {"res_fold": "bio_tcell", "max_rounds": 6},
    }

    comp_meth_folders_dict = {
        "cf10_3comp": [
            # Add folder paths here according to the label_list order.
        ],
        "cf10_8comp": [
            # Add folder paths here according to the label_list order.
        ],
        "kegg_4comp": [
            # Add folder paths here according to the label_list order.
        ],
        "sarcos_4comp": [
            # Add folder paths here according to the label_list order.
        ],
        "diamonds_4comp": [
            # Add folder paths here according to the label_list order.
        ],
        "bio_macrophage": [
            # Add folder paths here according to the label_list order.
        ],
        "bio_tcell": [
            # Add folder paths here according to the label_list order.
        ],
    }

    plot_neurips_strat(comp_meth_folders_dict, dataset_info, legend_only=args.legend_only, legend_outside=args.legend_outside)


if __name__ == '__main__':
    main()
