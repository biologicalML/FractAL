# import mplscience
import pandas as pd
import numpy as np
from tqdm import tqdm

np.random.seed(0)

inp_space = pd.read_csv("input_space_stim8.csv", index_col=0)
out_space = pd.read_csv("output_space_stim48.csv", index_col=0)

# prepare groundtruth for evaluation
# Metric 1: MSE Top k DE genes, requires ordered list of DE genes for each perturbation
index_list = out_space.columns.tolist()
ordered_de_dict = {}
# breakpoint()
for perturbation in tqdm(out_space.index):
    # sort the genes by their log fold change
    ordered_de = np.abs(out_space.loc[perturbation]).sort_values(ascending=False).index.tolist()[:100]
    ordered_de_dict[perturbation] = [index_list.index(gene) for gene in ordered_de]

# metric 2: binary AUPRC for perturbation, requires binary labels for each perturbation and gene
de_dict = {}
threshold = 0.5
for perturbation in tqdm(out_space.index):
    # get the log fold change for the perturbation
    lfc = out_space.loc[perturbation]
    mask = np.abs(lfc) > threshold
    de_dict[perturbation] = mask.astype(int).tolist()

X = inp_space
y = out_space

perturb_train = np.random.choice(X.index, size=int(X.shape[0] * 0.8), replace=False)
perturb_test = np.array(list(set(X.index) - set(perturb_train)))

X_train = X.loc[perturb_train]
X_test = X.loc[perturb_test]
y_train = y.loc[perturb_train]
y_test = y.loc[perturb_test]

X_new = pd.concat([X_train, X_test])
y_new = pd.concat([y_train, y_test])

ordered_de_list = []
de_mask_list = []
for perturbation in X_new.index:
    ordered_de_list.append(ordered_de_dict[perturbation])
    de_mask_list.append(de_dict[perturbation])

# breakpoint()
np.save("X.npy", X_new.to_numpy())
np.save("y.npy", y_new.to_numpy())
np.save("ordered_de_gt.npy", np.array(ordered_de_list))
np.save("de_mask_gt.npy", np.array(de_mask_list))
np.save("column_names.npy", y.columns.to_numpy())

