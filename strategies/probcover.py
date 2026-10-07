import numpy as np
import pandas as pd
import torch
from .interface import Strategy
from dataset import DictDataset, DataSplit

class ProbCover(Strategy):
    '''
    This is a coverage style strategy that chooses points which cover most of the space around it, 
    i.e. the one with highest number of edges. We borrow the official implementation of the strategy by
    its authors and adapt it to our code style
    '''

    def __init__(self, config, data: DictDataset, lab_idx: list[int], unlab_idx:list[int]):
        super().__init__(config, data)
        self.name = "probcover"
        self.lSet = lab_idx
        self.uSet = unlab_idx
        self.delta = config.probcover_delta
        self.use_trainer = True

    def construct_graph(self, logger, mtrainer, batch_size=500):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.

        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """
        self.relevant_indices = np.array(self.lSet + self.uSet)
        batch_size = min(batch_size, len(self.relevant_indices))
        if self.config.emb == "simclr":
            self.rel_features = self.load_simclr_embeds()[self.relevant_indices.tolist()]
        elif self.config.emb == "model":
            self.rel_features = mtrainer.get_probs_preds_and_repr(self.relevant_indices.tolist(), only_repr=True).cpu().numpy()
        elif self.config.emb == "input":
            self.rel_features = self.data.data.get_batch(self.relevant_indices.tolist())['X'].view(len(self.relevant_indices), -1).cpu().numpy()
        
        xs, ys, ds = [], [], []
        print(f'Start constructing graph using delta={self.delta}')
        logger.write(f'Start constructing graph using delta={self.delta}')
        # distance computations are done in GPU
        if torch.cuda.is_available():
            cuda_feats = torch.tensor(self.rel_features).cuda() #Shape is (N, D)
        else:
            cuda_feats = torch.tensor(self.rel_features) #Shape is (N, D)
        for i in range(len(self.rel_features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size] #Shape is (B, D)
            dist = torch.cdist(cur_feats, cuda_feats) #Shape is (B, N)
            mask = dist < self.delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + batch_size * i)
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        print(f'Finished constructing graph using delta={self.delta}')
        print(f'Graph contains {len(df)} edges.')
        logger.write(f'Graph contains {len(df)} edges.')
        self.graph_df = df

    def get_preds(self, logger, sample_per_round: int, round_num: int, **kwargs):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """
        mtrainer = kwargs.get('trainer', None)
        assert mtrainer is not None, "Model Trainer object is not passed to choose the active set"
        self.construct_graph(logger, mtrainer)
        selected = []

        # removing incoming edges to all covered samples from the existing labeled set
        alr_labeled = np.array([idx for idx, elem in enumerate(self.relevant_indices) if elem in self.lSet])

        edge_from_seen = np.isin(self.graph_df.x, alr_labeled)
        covered_samples = self.graph_df.y[edge_from_seen].unique()
        cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        
        for i in range(sample_per_round):
            coverage = len(covered_samples) / len(self.relevant_indices)
            # selecting the sample with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(self.relevant_indices))
            # cur = degrees.argmax()
            max_deg = np.flatnonzero(degrees == np.max(degrees))
            cur = max_deg[0]
            idx = 0
            while cur in selected and idx < len(max_deg): 
                idx += 1
                cur = max_deg[idx]
            
            if idx == len(max_deg):
                print("All points in the graph have been selected, stopping early.")
                raise ValueError("All points in the graph have been selected, stopping early.")

            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.append(cur)

        assert len(selected) == sample_per_round, 'added a different number of samples'
        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        logger.write(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Selected {len(activeSet)} new points to annotate for round {round_num}.\n')
        return activeSet.tolist(), remainSet.tolist()
