from .interface import Strategy
import numpy as np
from dataset import DictDataset, DataSplit

class Random(Strategy):
    """This strategy chooses the points to label randomly from the pool for 1 round."""

    def __init__(self, config, data: DictDataset | DataSplit, lab_idx: list[int], unlab_idx:list[int]):
        super().__init__(config, data)
        self.name = "random"
        self.lSet = lab_idx
        self.uSet = unlab_idx

    def get_preds(self, logger, sample_per_round: int, round_num: int, **kwargs):
        activeSet = np.random.choice(np.array(self.uSet), size=sample_per_round, replace=False)
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        logger.write(f"Selected {sample_per_round} new points to annotate for round {round_num}.\n")
        print(f"Selected {sample_per_round} new points to annotate for round {round_num}.\n")
        return activeSet.tolist(), remainSet.tolist()