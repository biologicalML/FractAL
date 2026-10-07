from .interface import Strategy
import numpy as np
from dataset import DictDataset, DataSplit
import torch
from sklearn.cluster import kmeans_plusplus

class BADGE(Strategy):
    """
    This strategy chooses diverse points using kmeans++ in a space where each point is embedded by the gradient it induces on 
    the last layer of the surrogate model. 
    """

    def __init__(self, config, data: DictDataset | DataSplit, lab_idx: list[int], unlab_idx:list[int]):
        super().__init__(config, data)
        self.lSet = lab_idx
        self.uSet = unlab_idx
        self.name = "badge"
        self.use_trainer = True

    def kmeans_plus_plus(self, X, k, round_num):
        if not self.config.dataset.startswith("cifar"):
            # Use sklearn kmeans++ since size is smaller
            return kmeans_plusplus(X, n_clusters=k, random_state=round_num)
        else:
            X = torch.from_numpy(X).to('cuda')
            n = X.shape[0]
            indices = []
            centers = []
            # first center
            idx = torch.randint(0, n, (1,), device=X.device)
            indices.append(idx.item())
            centers.append(X[idx])

            # initialize min distances
            min_dist = torch.cdist(X, centers[-1], p=2).squeeze() ** 2

            for _ in range(1, k):

                probs = min_dist / min_dist.sum()
                idx = torch.multinomial(probs, 1)
                indices.append(idx.item())
                centers.append(X[idx])

                # distance only to new center
                dist_new = torch.cdist(X, centers[-1], p=2).squeeze() ** 2

                # update running minimum
                min_dist = torch.minimum(min_dist, dist_new)

            return centers, np.array(indices)

    def get_grad_embeds_for_CE(self, mtrainer):
        ## Implemented only for classification tasks
        probs, labs, reprs = mtrainer.get_probs_preds_and_repr(self.uSet) #Probs: N  x num_class, labs: N x 1, reprs: N x K, weight: K x num_class
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, labs, 1)
        grad_embed = reprs.unsqueeze(2) * (probs - one_hot).unsqueeze(1) ## Grad for last layer is (y_pred - y) * input (outer product)
        grad_embeds = grad_embed.view(reprs.shape[0], -1)
        return grad_embeds.cpu().numpy()

    def get_preds(self, logger, sample_per_round: int, round_num: int, **kwargs):
        mtrainer = kwargs.get('trainer', None)
        assert mtrainer is not None, "Model Trainer object is not passed to choose the active set" 
        grad_embeds = self.get_grad_embeds_for_CE(mtrainer)
        print("Gradient Embeddings done. Beginning K-Means now")
        grad_embeds = grad_embeds.astype('float32') 
        _, chosen_idxs = self.kmeans_plus_plus(grad_embeds, sample_per_round, round_num)
        activeSet = np.array(self.uSet)[chosen_idxs.tolist()]

        """Add newly labeled data to lab_data and remove them from pool"""
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        logger.write(f"Selected {sample_per_round} new points to annotate for round {round_num}.\n")
        print(f"Selected {sample_per_round} new points to annotate for round {round_num}.\n")
        return activeSet.tolist(), remainSet.tolist()