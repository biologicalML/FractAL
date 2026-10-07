from .interface import Strategy
import numpy as np
from dataset import DictDataset,DataSplit
import torch

class MinMargin(Strategy):
    """This strategy chooses the points that are confused the most between its top 2 classes"""

    def __init__(self, config, data: DictDataset | DataSplit, lab_idx: list[int], unlab_idx:list[int]):
        super().__init__(config, data)
        self.lSet = lab_idx
        self.uSet = unlab_idx
        self.name = "minmargin"
        self.use_trainer = True

    def get_preds(self, logger, sample_per_round: int, round_num: int, **kwargs):
        mtrainer = kwargs.get('trainer', None)
        assert mtrainer is not None, "Model Trainer object is not passed to choose the active set" 
        unlab_probs = mtrainer.get_predictions(self.uSet, probs=True)
        vals, _ = unlab_probs.sort(dim=1, descending=True)
        vals = vals[:,:2]
        margins = vals[:,0] - vals[:,1]
        _, chosen_idxs = torch.topk(margins, sample_per_round, largest=False) #Get k elements with smallest margins
        activeSet = np.array(self.uSet)[chosen_idxs.tolist()]

        """Add newly labeled data to lab_data and remove them from pool"""
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        logger.write(f"Selected {sample_per_round} new points to annotate for round {round_num}.\n")
        print(f"Selected {sample_per_round} new points to annotate for round {round_num}.\n")
        return activeSet.tolist(), remainSet.tolist()