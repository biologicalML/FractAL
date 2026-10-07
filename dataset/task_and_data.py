import torch
import math
import numpy as np
from pathlib import Path
from typing import *
from os import PathLike
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from .custom_datasets import CIFAR10
from utils import set_seeds

def randperm(n: int, device: str = 'cpu') -> torch.Tensor:
    """
    Returns multiple random permutations.
    :param n_batch: Number of permutations.
    :param n: Length of permutations.
    :param device: PyTorch Device to put the permutations on.
    :return: Returns a torch.Tensor of integer type of shape [n_batch, n] containing the n_batch permutations.
    """
    return torch.randperm(n, device=device)

class DictDataset:
    """
    Represents a data set containing multiple tensors that can be accessed by their name (a string),
    for example {'x': inputs, 'y': targets}.
    """
    def __init__(self, tensors: Dict[str, torch.Tensor], device: str = None):
        """
        :param tensors: Dictionary of names and tensors.
        All tensors should have the same shape[0], i.e., the same number of samples.
        All tensors should have two dimensions,
        i.e., scalar targets have to be passed with a second dimension of shape 1.
        :param device: PyTorch device that the tensors should be moved to.
        If device is None, all tensors are moved to the device of the first tensor.
        """
        self.device = device if device is not None else next(iter(tensors.values())).device
        self.n_samples = next(iter(tensors.values())).shape[0]
        self.tensors = None if tensors is None else {key: t.to(device) for key, t in tensors.items()}

    def get_batch(self, idxs: list | torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns the tensors corresponding to the batch indexed by idxs.
        :param idxs: Tensor of indices to index the tensors of this object with.
        :return: Returns a dictionary {key: t[idxs, :] for key, t in self.tensors.items()}
        """
        if not isinstance(idxs,torch.Tensor):
            idxs = torch.tensor(idxs, dtype=torch.int64, device=self.device)
        return {key: t[idxs] for key, t in self.tensors.items()}

    def get_sub_dataset(self, idxs: torch.Tensor) -> 'DictDataset':
        """
        Returns a data set representing the batch given by the indices idxs.
        :param idxs: Tensor of indices used to index the tensors of this object with.
        :return: Returns the DictDataset with the subset of samples specified by idxs.
        """
        return DictDataset(self.get_batch(idxs), device=self.device)

    def __len__(self) -> int:
        """
        :return: Returns the number of samples of the tensors.
        """
        return self.n_samples

    def to(self, device: str) -> 'DictDataset':
        """
        Move all tensors to the given device.
        :param device: PyTorch Device.
        :return: Returns a DictDataset with all tensors moved to the given device.
        """
        return DictDataset(self.tensors, device=device)

class ParallelDictDataLoader:
    """
    This class enables vectorized data loading from DictDatasets,
    i.e., if multiple models are trained in parallel on the same data set,
    this class selects batches independently for each of the models.
    If vectorization is not needed, it is of course possible to use this class with only one model.
    """
    def __init__(self, ds: DictDataset, idxs: torch.Tensor, batch_size: int, shuffle: bool = False,
                 adjust_bs: bool = True, drop_last: bool = False):
        """
        :param dataset: A DictDataset from which tensors should be loaded
        :param idxs: Vectorized tensor of indices that specify a subset of ds from which data should be loaded.
        The tensor should have shape [n_samples],
        where n_samples is the number of samples that should be selected from ds.
        :param batch_size: default batch size, might be automatically adjusted
        :param shuffle: whether the dataset should be shuffled before each epoch
        :param adjust_bs: whether the batch_size may be lowered
        so that the batches are of more equal size while keeping the number of batches the same
        :param drop_last: whether the last batch should be omitted if it is smaller than the other ones
        """
        self.output_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.ds = ds.to(self.output_device)
        self.idxs = idxs.to(self.output_device)
        self.n_samples = idxs.shape[0]
        self.adjust_bs = adjust_bs
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.specified_batch_size = batch_size
        self.batch_size = min(batch_size, self.n_samples)

        if self.drop_last:
            self.n_batches = math.floor(self.n_samples / self.batch_size)
            if adjust_bs:
                self.batch_size = math.floor(self.n_samples / self.n_batches)
            self.sep_idxs = [self.batch_size * i for i in range(self.n_batches + 1)]
        else:
            self.n_batches = math.ceil(self.n_samples / self.batch_size)
            if adjust_bs:
                self.batch_size = math.ceil(self.n_samples / self.n_batches)
            self.sep_idxs = [self.batch_size * i for i in range(self.n_batches)] + [self.n_samples]

    def get_num_samples(self):
        """
        :return: Returns the number of samples that is sampled from (i.e. idxs.shape[1])
        """
        return self.n_samples

    def get_num_iterated_samples(self):
        """
        :return: Returns the number of samples that are visited in one epoch
        (might be less than get_num_samples() if drop_last=True).
        """
        if self.drop_last:
            return self.n_batches * self.batch_size
        return self.get_num_samples()

    def __len__(self):
        """
        :return: Returns the number of batches per epoch.
        """
        return self.n_batches

    def __iter__(self):
        """
        Allows to iterate over batches of an epoch.
        :return: Returns an iterator that allows to iterate over dictionaries of the form {name: tensor}
        with tensor.shape[0] <= batch_size.
        """
        if self.shuffle:
            perms = randperm(self.n_samples, device=self.ds.device)
            for start, stop in zip(self.sep_idxs[:-1], self.sep_idxs[1:]):
                batches = self.ds.get_batch(idxs=self.idxs[perms[start:stop]])
                yield batches['X'], batches['y']
        else:
            for start, stop in zip(self.sep_idxs[:-1], self.sep_idxs[1:]):
                batches = self.ds.get_batch(idxs=self.idxs[start:stop])
                yield batches['X'], batches['y']

class Task:
    """
    Represents a task, i.e., a data set and information what to do on the data set (how many batch AL steps etc).
    """
    def __init__(self,task_name: str, data_dir: str| PathLike, n_train: float, n_valid: float, n_test: float,
                 al_batch_sizes: List[int], task_type: str = "regr"):
        """
        Constructor. The actual data belonging to the task is loaded lazily, i.e., only when it is needed.
        :param data_info: DataInfo object representing the data set.
        :param task_name: Name of the task. Often same as name of dataset
        :param n_train: Number of initial training samples.
        :param n_valid: Number of validation samples.
        The remaining data_info.n_tvp - n_train - n_valid samples are used as initial pool samples.
        :param al_batch_sizes: List of batch sizes to acquire during batch active learning.
        """
        self.task_name = task_name
        self.data_dir = data_dir
        self.data = None  # load lazily
        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test = n_test
        self.al_batch_sizes = al_batch_sizes
        self.task_type = task_type

    def get_data(self) -> DictDataset:
        """
        :return: Returns a DictDataset containing the data set, with names 'X' for the inputs and 'y' for the targets.
        The tensor shape belonging to 'X' is [n_samples, n_features]
        and the tensor shape belonging to 'y' is [n_samples, 1].

        Handles if specific sizes are provided for init_train, val and test sets
        """
        if self.data is None:
            base_path = Path(self.data_dir) / self.task_name
            X = np.load(f'{base_path}/X.npy').astype('float32')
            if self.task_type == "clas":
                y = np.load(f'{base_path}/y.npy').astype('int64')
            else:
                y = np.load(f'{base_path}/y.npy').astype('float32')
            if y.ndim == 1:
                y = y[:, np.newaxis]
            self.data = DictDataset({'X': torch.as_tensor(X), 'y': torch.as_tensor(y)})
        self.n_samples = self.data.tensors['X'].shape[0]
        if not self.n_train.is_integer():
            self.n_train = int(self.n_train * self.n_samples)
        else:
            self.n_train = int(self.n_train)
        
        if not self.n_valid.is_integer():
            self.n_valid = min(int(self.n_valid * self.n_samples), 1024)
        else:
            self.n_valid = int(self.n_valid)

        if not self.n_test.is_integer():
            self.n_test = int(self.n_test * self.n_samples)
        else:
            self.n_test = int(self.n_test)
        
        self.n_pool = self.n_samples - self.n_train - self.n_valid - self.n_test
        return self.data

    @staticmethod
    def get_tabular_tasks(args):
        """
        Creates the task for the dataset
        """
        base_path = Path(args.data_dir)
        data_path = base_path / 'data'
        ds_name = args.dataset.lower()
        task = Task(ds_name, data_path, args.init_label_pct, args.val_pct, args.test_percent,
                            [args.al_batch_size] * args.num_al_steps, args.task_type)
        return task

class TaskSplit:
    """
    Represents one particular train-val-pool-test split of a task.
    It also preprocesses the task data according to the split.
    """
    def __init__(self, task: Task, use_pool_for_normalization: bool = True):
        """
        Creates the split and preprocesses the data. The preprocessed data set is stored in self.data.
        The idxs for init train, val, pool, test can be found in
        self.init_train_idxs, self.valid_idxs, self.pool_idxs, self.test_idxs.
        :param task: Task to split.
        :param id: Identifier of the split. Also serves as a seed for creating the split.
        :param use_pool_for_normalization: Whether to compute the statistics for centering and standardization
        only on the (initial) train set or on train+pool sets.
        """
        self.al_batch_sizes = task.al_batch_sizes
        self.data = task.get_data()
        self.task_name = task.task_name
        self.n_samples = self.data.tensors['X'].shape[0]
        self.use_pool_for_normalization = use_pool_for_normalization
        self.test_idxs = None

    def process_data(self, task: Task, seed_id: int):
        set_seeds(seed_id)
        self.test_idxs = np.arange(self.n_samples)[-task.n_test:]
        perm = np.random.permutation(self.n_samples)
        n_tvp = self.n_samples - task.n_test
        tvp_perm = np.array([el for el in perm if el not in self.test_idxs])
        
        s1 = task.n_train
        s2 = s1 + task.n_valid
        self.init_train_idxs = tvp_perm[:s1]
        self.valid_idxs = tvp_perm[s1:s2]
        self.pool_idxs = tvp_perm[s2:]

        # preprocess tensors
        X = self.data.tensors['X']
        y = self.data.tensors['y']
        if (not self.task_name.startswith("bio")):
            if self.use_pool_for_normalization:
                norm_idxs = np.concatenate([self.init_train_idxs, self.pool_idxs], axis=0)
            else:
                norm_idxs = self.init_train_idxs
            X_norm = X[norm_idxs]
            X = (X - X_norm.mean(dim=0, keepdim=True)) / (X_norm.std(dim=0, keepdim=True) + 1e-30)
            X = 5 * torch.tanh(0.2 * X)
        self.data = DictDataset({'X': X, 'y': self.data.tensors['y']})

    def get_data(self) -> DictDataset:
        """
        :return: Returns the DictDataset representing all of the data (train+val+pool+test)
        """
        return self.data

    def get_train_idxs(self) -> torch.Tensor:
        """
        :return: Returns the indices for the (initial) training data.
        """
        return self.init_train_idxs

    def get_valid_idxs(self) -> torch.Tensor:
        """
        :return: Returns the indices for the validation data.
        """
        return self.valid_idxs

    def get_pool_idxs(self) -> torch.Tensor:
        """
        :return: Returns the indices for the (initial) pool data.
        """
        return self.pool_idxs

    def get_test_idxs(self) -> torch.Tensor:
        """
        :return: Returns the indices for the test data.
        """
        return self.test_idxs

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, data, idxs: torch.Tensor, batch_size: int,
                 drop_last: bool = False, num_workers = 4):
        self.data = data
        subsetSampler = SubsetRandomSampler(idxs.cpu().numpy())
        super().__init__(dataset=data, num_workers=num_workers, batch_size=batch_size,
                                       sampler=subsetSampler, pin_memory=True, drop_last=drop_last)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class DataSplit:
    def __init__(self, cfg, train_data, test_data):
        self.data = train_data
        self.test_data = test_data
        self.n_train = cfg.init_label_pct
        self.n_valid = cfg.val_pct
        self.num_workers = 4
    
    def process_data(self, seed_id):
        """
        Given a seed, creates the train, test and val splits.
        """
        set_seeds(seed_id)
        self.makeLUVSets()
        self.test_idxs = np.arange(len(self.test_data))
    
    def makeLUVSets(self):
        """
        Initialize the labelled and unlabelled set by splitting the data into train
        and validation according to split_ratios arguments.

        Visually it does the following:

        |<------------- Train -------------><--- Validation --->

        |<--- Labelled --><---Unlabelled --><--- Validation --->
        
        OUTPUT:
        Sets the labelled, unlabelled set along with validation set
        """
        lSet = []
        uSet = []
        valSet = []
        
        n_dataPoints = len(self.data)
        all_idx = [i for i in range(n_dataPoints)]
        np.random.shuffle(all_idx)

        if not self.n_train.is_integer():
            train_splitIdx = int(self.n_train * self.n_samples)
        else:
            train_splitIdx = int(self.n_train)
        
        if not self.n_valid.is_integer():
            val_splitIdx = min(int(self.n_valid * self.n_samples), 1024)
        else:
            val_splitIdx = int(self.n_valid)
        lSet = all_idx[:train_splitIdx]
        valSet = all_idx[train_splitIdx:train_splitIdx + val_splitIdx]
        uSet = all_idx[train_splitIdx + val_splitIdx:]

        self.init_train_idxs = np.array(lSet, dtype=np.ndarray)
        self.pool_idxs = np.array(uSet, dtype=np.ndarray)
        self.valid_idxs = np.array(valSet, dtype=np.ndarray)

class Data:
    """
    Contains all data related functions.

    """
    def __init__(self, cfg):
        """
        Initializes dataset attribute of (Data class) object with specified "dataset" argument.
        INPUT:
        cfg: args parsed from command line
        """
        self.cfg = cfg
        self.dataset = cfg.dataset.upper()
        self.data_dir = Path(cfg.data_dir)/ 'data' / cfg.dataset
        self.aug_method = cfg.aug_method

    def about(self):
        """
        Show all properties of this class.
        """
        print(self.__dict__)

    def getPreprocessOps(self, is_train):
        """
        This function specifies the steps to be accounted for preprocessing.
        
        INPUT:
        None
        
        OUTPUT:
        Returns a list of preprocessing steps. Note the order of operations matters in the list.
        """
        ops = []
        norm_mean = []
        norm_std = []
        
        if self.dataset == "CIFAR10":
            ops = [transforms.RandomCrop(32, padding=4)]
            norm_mean = [0.4914, 0.4822, 0.4465]
            norm_std = [0.247 , 0.2435, 0.2616]
        else:
            raise NotImplementedError
        
        if (self.aug_method == 'hflip'):
            ops.append(transforms.RandomHorizontalFlip())

        if not is_train:
            ops = []
        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(norm_mean, norm_std))

        print("Preprocess Operations Selected ==> ", ops)
        return ops

    def getDataset(self, isTrain=True, isDownload=False):
        """
        This function returns the dataset instance and number of data points in it.
        
        INPUT:        
        isTrain (optional): Bool, If true then Train partition is downloaded else Test partition.
        
        isDownload (optional): Bool, If true then dataset is saved at path specified by "save_dir".
        
        OUTPUT:
        Returns the tuple of dataset instance and length of dataset.
        """
        
        preprocess_steps = self.getPreprocessOps(isTrain)
        if isTrain:
            test_preprocess_steps = preprocess_steps[-2:]
        else:
            test_preprocess_steps = preprocess_steps
        preprocess_steps = transforms.Compose(preprocess_steps)
        if self.dataset == "CIFAR10":
            cifar10 = CIFAR10(self.data_dir, train=isTrain, transform=preprocess_steps, test_transform=transforms.Compose(test_preprocess_steps), download=isDownload)
            return cifar10, len(cifar10)
        else:
            raise NotImplementedError