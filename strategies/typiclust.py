import numpy as np
import pandas as pd
import faiss
from sklearn.cluster import MiniBatchKMeans, KMeans
from .interface import Strategy
from dataset import DictDataset, DataSplit
import torch
import numpy as np
import faiss

def get_nn(features, num_neighbors, use_gpu=False):
    """
    Compute nearest neighbors using FAISS on CPU or GPU.

    Args:
        features (np.ndarray): shape (n, d)
        num_neighbors (int): number of neighbors (excluding self)
        use_gpu (bool): whether to use GPU

    Returns:
        distances, indices
    """
    features = features.astype(np.float32)
    d = features.shape[1]

    # Build CPU index
    index = faiss.IndexFlatL2(d)

    if use_gpu:
        # Move index to all available GPUs
        index = faiss.index_cpu_to_all_gpus(index)

    index.add(features)
    distances, indices = index.search(features, num_neighbors + 1)

    # drop self-neighbor
    return distances[:, 1:], indices[:, 1:]

def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False
    distances, indices = get_nn(features, num_neighbors, use_gpu=use_gpu)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance

def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality

def kmeans(features, num_clusters):
    if num_clusters <= 50:
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_


class TypiClust(Strategy):
    '''
    This strategy clusters the entire labeled + unlabeled pool (Except the val set).
    Too small clusters are dropped.

    Sort the clusters by the number of already labeled samples in them and then by cluster size in descending order.
    Choose the most typical points in the selected clusters.
    '''
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 20

    def __init__(self, config, data: DictDataset | DataSplit, lab_idx: list[int], unlab_idx:list[int]):
        super().__init__(config, data)
        self.name = "typiclust"
        self.lSet = lab_idx
        self.uSet = unlab_idx
        self.use_trainer = True
        self.clusters = None
        self.num_clusters = 0
        if self.config.dataset.startswith("bio"):
            self.MIN_CLUSTER_SIZE = 1 # Since the pool is small

    def init_features_and_clusters(self, mtrainer):
        num_clusters = min(len(self.lSet) + self.config.al_batch_size, self.MAX_NUM_CLUSTERS)
        print(f'Clustering into {num_clusters} clustering.')
        
        ## Condition to check to avoid clustering again in every iteration which is time consuming
        if self.config.emb == "model" or self.clusters is None or self.num_clusters != num_clusters:
            self.num_clusters = num_clusters
            self.relevant_indices = np.array(self.lSet + self.uSet)
            if self.config.emb == "simclr":
                self.features = self.load_simclr_embeds()[self.relevant_indices.tolist()]
            elif self.config.emb == "model":
                self.features = mtrainer.get_probs_preds_and_repr(self.relevant_indices.tolist(), only_repr=True).cpu().numpy()
            elif self.config.emb == "input":
                self.features = self.data.data.get_batch(self.relevant_indices.tolist())['X'].view(len(self.relevant_indices), -1).cpu().numpy()
            self.clusters = kmeans(self.features, num_clusters=num_clusters)
            print(f'Finished clustering into {num_clusters} clusters.')

    def get_preds(self, logger, sample_per_round: int, round_num: int, **kwargs):
        # using only labeled+unlabeled indices, without validation set.
        mtrainer = kwargs.get('trainer', None)
        assert mtrainer is not None, "Model Trainer object is not passed to choose the active set"
        self.init_features_and_clusters(mtrainer)
        
        features = self.features
        labels = np.copy(self.clusters)
        existing_indices = np.array([idx for idx in range(len(self.relevant_indices)) if self.relevant_indices[idx] in self.lSet], dtype=np.int64)#np.arange(len(self.lSet))
        assert len(existing_indices) == len(self.lSet)

        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids = np.arange(self.num_clusters)
        cluster_sizes = np.bincount(labels, minlength=len(cluster_ids))
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
                                    'neg_cluster_size': -1 * cluster_sizes})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size >= self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        labels[existing_indices] = -1
        
        selected = []
        for i in range(sample_per_round):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = (labels == cluster).nonzero()[0]
            shift = 0
            while len(indices) == 0:
                # all points in this cluster have been selected, move to next cluster
                print("Tried to select from empty cluster, moving to next cluster.")
                shift += 1
                cluster = clusters_df.iloc[(i+shift) % len(clusters_df)].cluster_id
                indices = (labels == cluster).nonzero()[0]
                
            rel_feats = features[indices]
            # in case we have too small cluster, calculate density among half of the cluster
            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]
            selected.append(idx)
            labels[idx] = -1

        selected = np.array(selected)
        assert len(selected) == sample_per_round, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]
        assert len(np.intersect1d(activeSet, self.lSet)) == 0, 'should be new samples'
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        
        logger.write(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Selected {len(activeSet)} new points to annotate for round {round_num}.\n')
        return activeSet.tolist(), remainSet.tolist()
