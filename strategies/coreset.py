from .interface import Strategy
import numpy as np
from dataset import DictDataset, DataSplit
import torch

class CoreSet(Strategy):
    """This strategy chooses the points with highest minimum distance to any other already labeled point"""

    def __init__(self, config, data: DictDataset | DataSplit, lab_idx: list[int], unlab_idx:list[int]):
        super().__init__(config, data)
        self.lSet = lab_idx
        self.uSet = unlab_idx
        self.name = "coreset"
        self.use_trainer = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def furthest_first(self, X, X_set, n):
        m = len(X)
        if len(X_set) == 0:
            min_dist = torch.from_numpy(np.tile(float("inf"), m)).to(self.device)
        else:
            dist_ctr = torch.cdist(X, X_set)
            min_dist, _ = torch.min(dist_ctr, dim=1)
        min_dist = min_dist.to(self.device)
        idxs = []

        for i in range(n):
            idx = torch.argmax(min_dist)
            idxs.append(idx.item())
            dist_new_ctr = torch.cdist(X, X[[idx], :]).to(self.device)
            min_dist = torch.minimum(min_dist, dist_new_ctr.squeeze(dim=-1))
            
        return idxs

    def get_preds(self, logger, sample_per_round: int, round_num: int, **kwargs):
        ## Handling different embedding types
        if self.config.emb == "simclr":
            embeds = self.load_simclr_embeds()
            chosen_idxs = self.furthest_first(
                X=torch.from_numpy(embeds[self.uSet]).view(len(self.uSet), -1).to(self.device),
                X_set=torch.from_numpy(embeds[self.lSet]).view(len(self.lSet), -1).to(self.device) if len(self.lSet) >0 else [],
                n=sample_per_round,
            )
        elif self.config.emb == "model":
            mtrainer = kwargs.get('trainer', None)
            assert mtrainer is not None, "Model Trainer object is not passed to choose the active set" 
            chosen_idxs = self.furthest_first(
                X=mtrainer.get_probs_preds_and_repr(self.uSet, only_repr=True).to(self.device),
                X_set=mtrainer.get_probs_preds_and_repr(self.lSet, only_repr=True).to(self.device) if len(self.lSet) >0 else [],
                n=sample_per_round,
            )
        elif self.config.emb == "input":
            chosen_idxs = self.furthest_first(
                X=self.data.data.get_batch(self.uSet)['X'].view(len(self.uSet), -1).to(self.device),
                X_set=self.data.data.get_batch(self.lSet)['X'].view(len(self.lSet), -1).to(self.device) if len(self.lSet) >0 else [],
                n=sample_per_round,
            )
        activeSet = np.array(self.uSet)[chosen_idxs]

        """Add newly labeled data to lab_data and remove them from pool"""
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        logger.write(f"Selected {sample_per_round} new points to annotate for round {round_num}.\n")
        print(f"Selected {sample_per_round} new points to annotate for round {round_num}.\n")
        return activeSet.tolist(), remainSet.tolist()