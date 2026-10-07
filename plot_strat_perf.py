import matplotlib.pyplot as plt
import numpy as np
import mplscience
import json
from args import read_args
import os

# from scipy import stats
from significance_test import paired_split_test, hierarchical_bootstrap_test

args = read_args()

strats = args.strategy
fnames = [
        ##<Add folder names here that you wish to plot and suitably update the label_list below
]

label_list = []

fig, ax = plt.subplots(figsize=(4, 3.3))
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

max_rounds = 6 if args.dataset.startswith("bio") else args.num_al_steps
x_vals = np.arange(max_rounds)[2:]
to_sub_val_means = None
to_sub_test_means = None
to_std_vals = None
offsets = np.linspace(-args.al_batch_size//10, args.al_batch_size//10, len(fnames))
if args.task_type == "clas":
    metric = 'acc' 
else:
    if args.dataset.startswith("bio"):
        metric = 'auprc'
    else:
        metric = 'loss'
with mplscience.style_context():
    for i, fname in enumerate(fnames):
        res_json = f"./results/{args.dataset}/{fname}/performance_records.json"
        y_val_split_vals = []
        y_test_split_vals = []
        try:
            with open(res_json) as f:
                res = json.load(f)
                for dsplit in range(args.n_data_splits):
                    res_split = res[f'dsplit_{dsplit+1}']
                    y_val_vals = []
                    y_test_vals = []
                    for run in range(args.n_runs):
                        
                        y_val = []
                        y_test = []
                        try:
                            for rd_dict in res_split[f'run_{run+1}']:
                                if "round" in list(rd_dict.keys()):
                                    if args.task_type == "clas":
                                        y_val.append(round(rd_dict['val_perf'][metric]*100, 4))
                                        y_test.append(round(rd_dict['test_perf'][metric]*100, 4))
                                    else:
                                        y_val.append(round(rd_dict['val_perf'][metric], 4))
                                        y_test.append(round(rd_dict['test_perf'][metric], 4))
                        except:
                            break
                        y_val_vals.append(y_val)
                        y_test_vals.append(y_test)
                    y_val_split_vals.append(y_val_vals)
                    y_test_split_vals.append(y_test_vals)

            ## Significance test between first two folders
            if not (i == 0):
                methb_vals = np.array(y_test_split_vals)[:,:,max_rounds-1]
                higher_is_better = (args.task_type=="clas" or args.dataset.startswith("bio"))
                res_dict = paired_split_test(metha_vals, methb_vals, higher_is_better=higher_is_better, test="ttest")
                print(f"T-Test: {res_dict}")
                hie_dict = hierarchical_bootstrap_test(metha_vals, methb_vals, higher_is_better=higher_is_better)
                print(f"Hierarchical Bootstrap Test: {hie_dict}")
            else:
                metha_vals = np.array(y_test_split_vals)[:,:,max_rounds-1]
            
            y_val_split_vals = np.array(y_val_split_vals).mean(axis=0) #Shape is (n_splits, n_runs, n_steps)
            y_test_split_vals = np.array(y_test_split_vals).mean(axis=0) #Shape is (n_splits, n_runs, n_steps)
            y_val_means = y_val_split_vals.mean(axis=0).tolist()
            y_val_err = y_val_split_vals.std(axis=0).tolist()
            y_test_means = y_test_split_vals.mean(axis=0).tolist()
            y_test_err = y_test_split_vals.std(axis=0).tolist()
        except:
            print(f"Results for {label_list[i]} on {args.dataset} not found at {res_json}!")
            continue
        print(f"{label_list[i]}:\nTest Mean: {y_test_means[max_rounds-1]}\nTest Err: {y_test_err[max_rounds-1]/np.sqrt(args.n_runs)}")
        
        if i==0:
            to_sub_val_means = y_val_means
            to_sub_test_means = y_test_means
            to_std_vals = y_test_split_vals
            to_std_val_vals = y_val_split_vals
        y_test_means = np.array(y_test_means) - np.array(to_sub_test_means)
        y_val_means = np.array(y_val_means) - np.array(to_sub_val_means)
        y_test_err = np.std(np.array(y_test_split_vals) - np.array(to_std_vals)[:len(y_test_split_vals),:], axis=0)
        y_val_err = np.std(np.array(y_val_split_vals) - np.array(to_std_val_vals)[:len(y_val_split_vals),:], axis=0)
        color = color_list[label_list[i]] if label_list[i] in color_list else None
        if color is not None:
            line, = ax.plot(x_vals, y_test_means[2:max_rounds], label=label_list[i], lw=2.5, marker="o", markersize=4, color=color)
        else:
            line, = ax.plot(x_vals, y_test_means[2:max_rounds], label=label_list[i], lw=2.5, marker="o", markersize=4)
        # Use the color of the line for the shaded area
        ax.fill_between(
            x_vals, 
            y_test_means[2:max_rounds] - y_test_err[2:max_rounds], 
            y_test_means[2:max_rounds] + y_test_err[2:max_rounds], 
            color=line.get_color(), 
            alpha=0.15  # Keep alpha low for overlapping groups
        )
    ax.set_xticks(x_vals)
    ax.set_xlabel('AL Iteration', fontsize=LABEL_SIZE)
    if args.task_type == "clas":
        ax.set_ylabel(r'Acc. Improvement', fontsize=LABEL_SIZE)
    else:
        if args.dataset.startswith("bio"):
            ax.set_ylabel(r'AUPRC Improvement', fontsize=LABEL_SIZE)
        else:
            ax.set_ylabel(r'Relative MSE$\downarrow$', fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)

    # Legend handling: inside, outside, or saved separately
    handles, labels = ax.get_legend_handles_labels()
    plot_dir = os.path.join("plots", args.dataset)
    os.makedirs(plot_dir, exist_ok=True)

    if args.legend_only:
        # Save legend as a separate figure and do not draw it on the main plot
        if len(labels) == 0:
            print("No legend entries to save.")
        else:
            # Create a dedicated figure for the legend
            # Make height proportional to number of labels
            n_labels = len(labels)
            legend_fig = plt.figure(figsize=(3, max(1, 0.4 * n_labels)))
            legend = legend_fig.legend(handles, labels, loc='center', frameon=False, fontsize=LEGEND_SIZE)
            legend_fig.tight_layout()
            legend_path_pdf = os.path.join(plot_dir, f'legend_{len(labels)}items.pdf')
            legend_path_png = os.path.join(plot_dir, f'legend_{len(labels)}items.png')
            legend_fig.savefig(legend_path_pdf, dpi=300, bbox_inches='tight')
            legend_fig.savefig(legend_path_png)
            plt.close(legend_fig)
            print(f"Saved legend to {legend_path_pdf}")
        # Save main plot without legend
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/{label_list}_{len(label_list)}comp_test_{metric}{'_coarse' if args.coarse else ""}.png')
        plt.savefig(f'{plot_dir}/{label_list}_{len(label_list)}comp_test_{metric}{'_coarse' if args.coarse else ""}.pdf', dpi=300, bbox_inches='tight')
    else:
        if args.legend_outside:
            # Place legend to the right outside the axes
            if len(labels) > 0:
                ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=LEGEND_SIZE)
            # leave space on the right when saving
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        else:
            # Default: inside the plot (upper right)
            if len(labels) > 0:
                ax.legend(loc='center right', frameon=True, fontsize=LEGEND_SIZE)
            plt.tight_layout()

        plt.savefig(f'{plot_dir}/{label_list}_{len(label_list)}comp_test_{metric}{'_coarse' if args.coarse else ""}.png')
        plt.savefig(f'{plot_dir}/{label_list}_{len(label_list)}comp_test_{metric}{'_coarse' if args.coarse else ""}_full.pdf', dpi=300, bbox_inches='tight')