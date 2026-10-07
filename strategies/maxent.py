from .interface import Strategy
import numpy as np
from dataset import DictDataset, DataSplit
import torch
from scipy.stats import entropy

class MaxEnt(Strategy):
    """This strategy chooses the points that have highest entropy amongst its logits."""

    def __init__(self, config, data: DictDataset|DataSplit, lab_idx: list[int], unlab_idx:list[int]):
        super().__init__(config, data)
        self.lSet = lab_idx
        self.uSet = unlab_idx
        self.name = "maxent"
        self.use_trainer = True

    def get_preds(self, logger, sample_per_round: int, round_num: int, **kwargs):
        mtrainer = kwargs.get('trainer', None)
        assert mtrainer is not None, "Model Trainer object is not passed to choose the active set" 
        unlab_probs = mtrainer.get_predictions(self.uSet, probs=True).numpy()
        entropies = torch.from_numpy(entropy(unlab_probs, axis=1))
        _, chosen_idxs = torch.topk(entropies, sample_per_round) #Get k elements with largest entropies
        activeSet = np.array(self.uSet)[chosen_idxs.tolist()]

        """Add newly labeled data to lab_data and remove them from pool"""
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        logger.write(f"Selected {sample_per_round} new points to annotate for round {round_num}.\n")
        print(f"Selected {sample_per_round} new points to annotate for round {round_num}.\n")
        return activeSet.tolist(), remainSet.tolist()