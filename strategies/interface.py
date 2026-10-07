from abc import ABC, abstractmethod
from dataset import DictDataset, DataSplit
from pathlib import Path
import numpy as np
import os

class Strategy(ABC):
    """This is the interface that all strategies in this package will implement.
    It can be imported for type annotations."""

    def __init__(self, config, data: DictDataset | DataSplit):
        self.data = data
        self.config = config
        self.use_trainer = False

    def load_simclr_embeds(self):
        data_dir = Path(self.config.data_dir) / "data" / self.config.dataset
        embeds = np.load(os.path.join(data_dir, "cifar10_train_simclr.npy"))
        return embeds

    def update_lab_and_unlab_sets(cls, new_lab_idxs: list[int], uSet:list[int]):
        cls.lSet = list(set(cls.lSet).union(set(new_lab_idxs)))
        cls.uSet = uSet

    @abstractmethod
    def get_preds(self, logger, sample_per_round: int, round_num: int, **kwargs):
        """Return the round wise performance of the strategy."""