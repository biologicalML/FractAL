import scanpy as sc
import pandas as pd
import numpy as np

from tqdm import tqdm

np.random.seed(0)

# load LFC pseq data
lfc_pseq = pd.read_csv("mocavi_de_exp.csv").pivot(index='group1', columns='Unnamed: 0', values='lfc_median')
lfc_pseq = lfc_pseq.loc[~lfc_pseq.index.str.contains("NTC|Olfr|Control")].copy() 

# # load ops bulk data
bulk_ops = sc.read_h5ad('adata_mean_genes_ops.h5ad')

# check for intersection of rows
pseq_genes = lfc_pseq.index
ops_genes = bulk_ops.obs_names
print("Number of genes in pseq:", len(pseq_genes))
print("Number of genes in ops:", len(ops_genes))
print("Number of genes in common:", len(set(pseq_genes) & set(ops_genes)))
common_genes = list(set(pseq_genes) & set(ops_genes))

bulk_ops = bulk_ops[common_genes].copy()
lfc_pseq = lfc_pseq.loc[common_genes].copy()

# filter for expressed genes of interest (the ones changing per perturbation)
#read the file data/gex_cluster_dict.json
import json
with open("gex_cluster_dict.json", "r") as f:
    gex_cluster_dict = json.load(f)
# flatten the dictionary and get the unique genes
expressed_genes = set()
for cluster, genes in gex_cluster_dict.items():
    expressed_genes.update(genes)
expressed_genes = list(expressed_genes)
print("Number of expressed genes:", len(expressed_genes))
# filter the dataframes for the expressed genes
lfc_pseq = lfc_pseq[expressed_genes].copy()

# prepare groundtruth for evaluation
# Metric 1: MSE Top k DE genes, requires ordered list of DE genes for each perturbation
index_list = lfc_pseq.columns.tolist()
ordered_de_dict = {}

for perturbation in tqdm(lfc_pseq.index):
    # sort the genes by their log fold change
    ordered_de = np.abs(lfc_pseq.loc[perturbation]).sort_values(ascending=False).index.tolist()[:100]
    ordered_de_dict[perturbation] = [index_list.index(gene) for gene in ordered_de]

# metric 2: binary AUPRC for perturbation, requires binary labels for each perturbation and gene
de_dict = {}
threshold = 0.5
for perturbation in tqdm(lfc_pseq.index):
    # get the log fold change for the perturbation
    lfc = lfc_pseq.loc[perturbation]
    mask = np.abs(lfc) > threshold
    de_dict[perturbation] = mask.astype(int).tolist()


X = pd.DataFrame(bulk_ops.X, index=bulk_ops.obs_names, columns=bulk_ops.var_names)
y = lfc_pseq

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

np.save("X.npy", X_new.to_numpy())
np.save("y.npy", y_new.to_numpy())
np.save("ordered_de_gt.npy", np.array(ordered_de_list))
np.save("de_mask_gt.npy", np.array(de_mask_list))
np.save("column_names.npy", y.columns.to_numpy())

