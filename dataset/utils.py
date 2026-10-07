import numpy as np

def read_bio_gt(dataset):
    topk_gt = np.load(f"./dataset/data/{dataset}/ordered_de_gt.npy").tolist()
    de_mask_gt = np.load(f"./dataset/data/{dataset}/de_mask_gt.npy")
    column_names = np.load(f"./dataset/data/{dataset}/column_names.npy", allow_pickle=True).tolist()

    return topk_gt, de_mask_gt, column_names