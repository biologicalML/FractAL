import numpy as np
import torch
from torch import nn
from dataset import ParallelDictDataLoader, MultiEpochsDataLoader, TaskSplit, read_bio_gt
from utils import log, set_seeds
from . import utils
from .models import MLP, ResNet
from pathlib import Path
from .dataset_model_dict import dm_dict
from .resnet import BasicBlock, load_pretrained
import torch.nn.functional as F
import copy
from sklearn.metrics import average_precision_score

class ModelTrainer:
    def __init__(self, data, args):
        self.args = args
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
        if isinstance(data, TaskSplit):
            self.data = data.data
        else:
            self.data = data #Either DictDataset or DataSplit
        self.seed = args.id
        self.res_dir = args.res_dir
        self.best_valid_mses = np.inf
        self.best_valid_acc = 0
        self.use_lr_sched = args.use_lr_sched
        self.act_fn = args.model_actvn
        self.best_epoch = 0

    def get_model(self, logger: log.InstanceLogger, round_num: int):
        ## Initialize the surrogate model
        rec = dm_dict[self.args.dataset]
        assert rec["type"] == self.args.model, f"Config for {self.args.dataset} and {self.args.model} does not exist in the dictionary"
        if round_num == 0:
            set_seeds(self.seed)
        if rec["type"] == "mlp":
            n_features = self.data.tensors['X'].shape[1]
            self.model = MLP(n_features=n_features, act = self.act_fn, **rec["kwargs"])
        elif rec["type"] == "resnet18":
            self.model = ResNet(BasicBlock, use_dropout=self.args.dropout, **rec["kwargs"])
            if self.args.pretrained:
                self.model = load_pretrained(self.model, 'resnet18')
        else:
            raise NotImplementedError(f"Model {rec["type"]} not recognized")
        self.model = self.model.to(self.device)        
        self.save_or_load_model(round_num=round_num)

    def save_or_load_model(self, round_num=-1, from_scratch=False, opt=None, retrain=False):
        '''
        Save or load the model checkpoint and the optimizer checkpoint as well
        '''
        if not retrain:
            if round_num == 0:
                torch.save(self.model.state_dict(), Path(self.res_dir) / f'model_init_rd0.pt')
            else:
                self.model.load_state_dict(torch.load(Path(self.res_dir) / f'model_init_rd0.pt'))
        else:
            if from_scratch:
                self.model.load_state_dict(torch.load(Path(self.res_dir) / f'model_init_rd0.pt'))
            else:
                ckpt = torch.load(Path(self.res_dir) / 'best_model.pt', weights_only=False)
                self.model.load_state_dict(ckpt['model_state_dict'])
                opt.load_state_dict(ckpt['optimizer'])
                lr = opt.param_groups[0]['lr']
                return opt, lr

    def train(self, logger: log.InstanceLogger, train_idxs:list[int], valid_idxs: list[int], round_num:int):
        train_timer = utils.Timer()
        self.get_model(logger, round_num)
        ## Choose the appropriate loss function
        self.loss_fn = nn.CrossEntropyLoss(reduction='none') if self.args.task_type == "clas" else nn.MSELoss(reduction='none')
        train_timer.start()
        ## Determine the type of learning rate scheduler to use
        lr_sched = 'cos' if self.args.dataset.startswith("cifar") else 'lin'
        #Train the model on the current train set
        self.fit_model(logger, train_idxs, valid_idxs, weight_decay=self.args.wd, n_epochs=self.args.train_epochs, batch_size=self.args.train_batch_size, lr = self.args.lr, lr_sched=lr_sched)
        train_timer.pause()
        
        logger.write(f"Training completed for round {round_num} in time {train_timer.acc_time_total}\n")
        print(f"Training completed for round {round_num} in time {train_timer.acc_time_total}\n")

    def get_opt(self, weight_decay):
        ## Initialize the optimizer
        if self.args.opt_name == 'sgd':
            opt = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=weight_decay,nesterov=True)
        else:
            if weight_decay > 0.0:
                if self.use_lr_sched:
                    opt = torch.optim.AdamW(self.model.parameters(), weight_decay=weight_decay)
                else:
                    opt = torch.optim.AdamW(self.model.parameters(), weight_decay=weight_decay, lr = self.args.lr)
            else:
                if self.use_lr_sched:
                    opt = torch.optim.Adam(self.model.parameters())
                else:
                    opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        return opt

    def get_epoch_lr(self,cur_epoch, lr_sched, n_epochs):
        ## Get the learning rate at current epoch according to the scheduling policy
        if lr_sched == 'lin':
            lr = 1.0 - cur_epoch / n_epochs
        elif lr_sched == 'cos':
            lr = 0.5 * (1.0 + np.cos(np.pi * cur_epoch / n_epochs))
        else:
            raise ValueError(f'Unknown lr sched "{lr_sched}"')
        return lr

    def prepare_loader(self,idxs, batch_size, drop_last, is_test, is_val=False):
        ## Prepare the data loaders with respective params
        idxs = torch.tensor(idxs, dtype=torch.int64, device=self.device)
        if self.args.dataset.startswith("cifar"):
            if is_test:
                data = self.data.test_data
            elif is_val:
                data = self.data.data
                data.val_mode = True
            else:
                data = self.data.data
                data.val_mode = False
            dl = MultiEpochsDataLoader(data, idxs, batch_size=batch_size, drop_last=drop_last,num_workers=4)
        else:
            data = self.data
            dl = ParallelDictDataLoader(data, idxs, batch_size=batch_size, shuffle=False,
                                        adjust_bs=False, drop_last=drop_last)
        return dl    

    def run_validation(self, valid_dl,num_val_examples):
        ## Run the current model on the validation set
        valid_sses = 0
        acc = 0
        self.model.eval()
        n_out_dims = 0
        with torch.no_grad():
            for batch in valid_dl:
                (X, y) = batch
                X,y = X.to(self.device), y.to(self.device)
                if self.args.dataset.startswith("bio") and n_out_dims == 0:
                    n_out_dims = y.shape[1]
                if len(y.shape) != 1:
                    y = y.squeeze(dim=-1)
                y_pred = self.model(X)
                valid_sses = valid_sses + self.loss_fn(y_pred.squeeze(dim=-1), y).sum()
                if self.args.task_type == "clas":
                    preds = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1)
                    b_acc = (preds == y)
                    acc += b_acc.sum().item()
            
        valid_mses = (valid_sses / num_val_examples).detach().cpu().numpy()
        acc = acc/num_val_examples
        if self.args.dataset.startswith("bio"):
            valid_mses = valid_mses/n_out_dims
        
        return valid_mses, acc
    
    def fit_model(self, logger, train_idxs, valid_idxs, save_model=True, n_epochs=15, batch_size=4, lr=1e-1, weight_decay=0.0,**config):
        print(f"Size of train data is {len(train_idxs)} now")
        do_valid = valid_idxs is not None and len(valid_idxs) > 0
        valid_dl = None if not do_valid else self.prepare_loader(valid_idxs, batch_size,drop_last=False,is_test=False, is_val=True)
        train_dl = self.prepare_loader(train_idxs, batch_size, drop_last=True,is_test=False)
        self.best_valid_mses = np.inf
        self.best_valid_acc = 0
        
        opt = self.get_opt(weight_decay)
        
        lr_sched = config.get('lr_sched', 'lin')
        for i in range(n_epochs):
            # do one training epoch
            step = 0
            self.model.train()

            ### UPDATE THE LEARNING RATE ACCORDING TO SCHEDULE
            if self.use_lr_sched:
                if self.args.dataset.startswith("bio"):
                    epoch_lr = max(self.get_epoch_lr(i,lr_sched,n_epochs) * lr, 0.1)
                else:
                    epoch_lr = self.get_epoch_lr(i,lr_sched,n_epochs) * lr
                for group in opt.param_groups:
                    group['lr'] = epoch_lr

            #Set val mode to false for the dataloader
            if self.args.dataset.startswith("cifar"):
                train_dl.dataset.val_mode = False
        
            for batch in train_dl:
                opt.zero_grad()
                (X, y) = batch
                X,y = X.to(self.device), y.to(self.device)
        
                y_pred = self.model(X)  # shape: batch_size x 1
        
                loss = self.loss_fn(y_pred.squeeze(dim=-1), y.squeeze(dim=-1)).mean()
                loss.backward()
                if step % 4 == 0:
                    print(f"Epoch {i}: Loss {loss.item()}")
                opt.step()
                step += 1
            if do_valid:
                # do one valid epoch
                if self.args.dataset.startswith("cifar"):
                    valid_dl.dataset.val_mode = True
                valid_mses,valid_acc = self.run_validation(valid_dl,len(valid_idxs))
                if self.args.task_type == "regr":
                    if valid_mses.item() < self.best_valid_mses:
                        self.best_valid_mses = valid_mses.item()
                        self.best_epoch = i
                        if save_model:
                            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer': opt.state_dict()}, Path(self.res_dir) / 'best_model.pt')
                else:
                    if valid_acc >= self.best_valid_acc:
                        self.best_valid_mses = valid_mses.item()
                        self.best_valid_acc = valid_acc
                        self.best_epoch = i
                        if save_model:
                            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer': opt.state_dict()}, Path(self.res_dir) / 'best_model.pt')
                
                if i == 0 or (i+1)%5 == 0:
                    metric = 'Loss' if self.args.task_type == "regr" else 'Acc'
                    value = valid_mses if self.args.task_type == "regr" else valid_acc
                    logger.write(f"Epoch: {i} Valid {metric}: {value}")
                    print(f"Epoch: {i} Valid {metric}: {value}")

        if do_valid and save_model:
            ckpt = torch.load(Path(self.res_dir) / 'best_model.pt', weights_only=False)
            self.model.load_state_dict(ckpt['model_state_dict'])
        elif save_model:
            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer': opt.state_dict()}, Path(self.res_dir) / 'best_model.pt')

    def get_gradients(self, data_idxs, last_layer=False, aggr=True):
        bs = self.args.train_batch_size if aggr else 1
        data_dl = self.prepare_loader(data_idxs, bs, drop_last=False, is_test=False, is_val=True)

        self.model.eval()
        if last_layer:
            for param in self.model.model.parameters():
                param.requires_grad = False

        if aggr:
            # Accumulate directly instead of storing per-batch grads and summing later.
            accum = None
            total_n = 0
            for batch in data_dl:
                self.model.zero_grad(set_to_none=True)
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                if len(y.shape) != 1:
                    y = y.squeeze(dim=-1)
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred.squeeze(dim=-1), y).mean()
                loss.backward()

                params = self.model.last_layer.parameters() if last_layer else self.model.parameters()
                batch_grads = [p.grad.detach() * len(X) for p in params]

                if accum is None:
                    accum = batch_grads  # first batch: take ownership, no clone needed yet
                else:
                    for i, g in enumerate(batch_grads):
                        accum[i] += g  # in-place accumulation, no intermediate list-of-lists
                total_n += len(X)

            self.model.zero_grad(set_to_none=True)
            for param in self.model.parameters():
                param.requires_grad = True

            return [g / total_n for g in accum]

        else:
            # Per-sample grads still need to be returned individually — keep as-is.
            grads = []
            for batch in data_dl:
                self.model.zero_grad(set_to_none=True)
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                if len(y.shape) != 1:
                    y = y.squeeze(dim=-1)
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred.squeeze(dim=-1), y).mean()
                loss.backward()

                params = self.model.last_layer.parameters() if last_layer else self.model.parameters()
                batch_grads = [p.grad.clone().detach() * len(X) for p in params]
                grads.append(batch_grads)

            self.model.zero_grad(set_to_none=True)
            for param in self.model.parameters():
                param.requires_grad = True

            return grads

    def get_predictions(self, data_idxs, detach=True, probs=False, is_test=False, is_val=False):
        bs = self.args.train_batch_size
        data_dl = self.prepare_loader(data_idxs,bs,drop_last=False,is_test=is_test, is_val=(not is_test))
        all_preds = []
        for param in self.model.parameters():
            param.grad = None
        self.model.eval()
        for batch in data_dl:
            (X,y) = batch
            X,y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)  # shape: batch_size x 1
            if self.args.task_type == "clas":
                y_pred == nn.Softmax(dim=1)(y_pred)
                if not probs:
                    y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            if detach:
                all_preds.append(y_pred.detach().cpu().clone())
            else:
                all_preds.append(y_pred)
            
        all_preds = torch.cat(all_preds, dim=0)  # shape: num_data x 1
        return all_preds  # torch tensor of shape num_data x 1
    
    def get_probs_preds_and_repr(self, data_idxs, only_repr = False):
        bs = self.args.train_batch_size
        data_dl = self.prepare_loader(data_idxs,bs,drop_last=False,is_test=False, is_val=True)

        all_preds = []
        all_labs = []
        all_repr = []
        for param in self.model.parameters():
            param.grad = None
        self.model.eval()
        for batch in data_dl:
            (X,y) = batch
            X,y = X.to(self.device), y.to(self.device)
            y_pred, repr = self.model(X, repr=True)  # shape: batch_size x 1
            if self.args.task_type == "clas":
                y_pred == nn.Softmax(dim=1)(y_pred)
                y_lab = torch.argmax(y_pred, dim=1, keepdim=True)
                all_labs.append(y_lab.detach().clone())
            
            all_preds.append(y_pred.detach().clone())
            all_repr.append(repr.detach().clone())

        all_repr = torch.cat(all_repr, dim=0)  # shape: num_data x 1    
        all_preds = torch.cat(all_preds, dim=0)  # shape: num_data x 1
        if self.args.task_type == "clas":
            all_labs = torch.cat(all_labs, dim=0)  # shape: num_data x 1
        
        if only_repr:
            return all_repr
        elif self.args.task_type == "regr":
            return all_preds, all_repr 
        else:
            return all_preds, all_labs, all_repr

    def get_repr_and_labels(self, data_idxs):
        """Run one eval-mode, no_grad pass over data_idxs and return the frozen
        penultimate-layer representation (input to last_layer) plus true labels.
        Used to avoid re-running the full backbone forward pass repeatedly when
        only last_layer parameters are being solved for (e.g. inside CG)."""
        bs = self.args.train_batch_size
        data_dl = self.prepare_loader(data_idxs, bs, drop_last=False, is_test=False, is_val=True)

        all_repr = []
        all_y = []
        self.model.eval()
        with torch.no_grad():
            for batch in data_dl:
                (X, y) = batch
                X, y = X.to(self.device), y.to(self.device)
                if len(y.shape) != 1:
                    y = y.squeeze(dim=-1)
                _, repr = self.model(X, repr=True)
                all_repr.append(repr.detach().clone())
                all_y.append(y.detach().clone())

        all_repr = torch.cat(all_repr, dim=0)
        all_y = torch.cat(all_y, dim=0)
        return all_repr, all_y

    def get_total_loss(self, data_idxs, is_test=False, is_val=False):
        bs = self.args.train_batch_size
        data_dl = self.prepare_loader(data_idxs, bs, drop_last=False, is_test=is_test, is_val=(not is_test))
        total_loss = None
        for param in self.model.parameters():
            param.grad = None
        self.model.train()
        for batch in data_dl:
            (X, y) = batch
            X,y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)  # shape: batch_size x 1
            if len(y.shape) != 1:
                y = y.squeeze(dim=-1)
            if self.args.dataset.startswith("bio"):
                loss = self.loss_fn(y_pred.squeeze(dim=-1), y).mean(dim=-1).sum()
            else:    
                loss = self.loss_fn(y_pred.squeeze(dim=-1), y).sum()
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
        return total_loss/len(data_idxs)  # float scalar

    def get_topk_mse(self, preds, y, topk_gt):
        topk_mse = np.array([0]*len(self.args.topk_mse))
        for (model_pred, gt, topk) in zip(preds, y, topk_gt):
            for i, kval in enumerate(self.args.topk_mse):
                topk_sel = topk[:kval]
                topk_mse[i] += ((model_pred[topk_sel] - gt[topk_sel])**2).mean().item()
        return topk_mse
    
    def get_auprc(self, preds, de_mask_gt):
        pred_flat = preds.numpy().flatten()
        de_mask_flat = de_mask_gt.flatten()
        return average_precision_score(de_mask_flat, np.abs(pred_flat))

    def test_model(self, logger, test_idxs, al_step, if_val = False, if_test=False, if_retrain=False):
        if len(test_idxs) == 0:
            if self.args.task_type == "regr":
                return {'loss': np.inf}
            else:
                return {'loss': np.inf, 'acc': 0.0}
        if self.args.dataset.startswith("bio"):
            topk_gt, de_mask_gt, column_names = read_bio_gt(self.args.dataset)
        
        bs = self.args.train_batch_size
        test_dl = self.prepare_loader(test_idxs,bs, drop_last=False,is_test=if_test, is_val=(not if_test))
        
        test_loss = 0
        acc = 0
        topk_mse_loss = np.array([0]*len(self.args.topk_mse))
        preds_list = []
        n_outdims = 0
        with torch.no_grad():
            self.model.eval()
            for i, batch in enumerate(test_dl):
                (X, y) = batch
                X,y = X.to(self.device), y.to(self.device)
                if self.args.dataset.startswith("bio") and n_outdims == 0:
                    n_outdims = y.shape[1]
                if len(y.shape) != 1:
                    y = y.squeeze(dim=-1)
                y_pred = self.model(X)
                
                test_loss = test_loss + self.loss_fn(y_pred.squeeze(dim=-1), y).sum()
                if self.args.task_type == "clas":
                    preds = torch.argmax(nn.Softmax(dim=1)(y_pred), dim=1)
                    b_acc = (preds == y)
                    acc += b_acc.sum().item()
                if self.args.dataset.startswith("bio"):
                    topk_mse_loss += self.get_topk_mse(y_pred.cpu(), y.cpu(), [topk_gt[idx] for idx in test_idxs[i*bs:min((i+1)*bs, len(test_idxs))]])
                preds_list.append(y_pred.cpu())
        acc = acc/len(test_idxs)
        test_loss = (test_loss / len(test_idxs)).item()

        if self.args.dataset.startswith("bio"):
            auprc_score = self.get_auprc(torch.cat(preds_list), de_mask_gt[test_idxs])
            test_loss = test_loss/n_outdims
            topk_mse_norm = [elem/len(test_idxs) for elem in topk_mse_loss.tolist()]
        
        if if_val:
            print(f'Val results: Loss={test_loss} {f"Acc.={acc}" if self.args.task_type == "clas" else ""}')
            logger.write(f'After AL Step {al_step}: Val results: Loss={test_loss} {f"Acc.={acc}" if self.args.task_type == "clas" else ""}\n')
        elif if_test:
            print(f'Test results: Loss={test_loss} {f"Acc.={acc}" if self.args.task_type == "clas" else ""}')
            logger.write(f'After AL Step {al_step}: Test results: Loss={test_loss} {f"Acc.={acc}" if self.args.task_type == "clas" else ""}\n')
        elif not if_retrain:
            logger.write(f'After AL Step {al_step}: Train results: Loss={test_loss} {f"Acc.={acc}" if self.args.task_type == "clas" else ""}\n')
        
        if self.args.dataset.startswith("bio"):
            print(f'Results: Loss={test_loss} Top-{self.args.topk_mse} MSE={topk_mse_norm} AUPRC={auprc_score} {f"Acc.={acc}" if self.args.task_type == "clas" else ""}')
            return {'loss': test_loss, 'acc': acc, 'topk_mse': topk_mse_norm, 'auprc': auprc_score}
        elif self.args.task_type == "regr" and (not self.args.dataset.startswith("bio")):
            return {'loss': test_loss}
        else:
            return {'loss': test_loss, 'acc': acc}
    
    def get_all_params(self, last_layer=False):
        all_params = []
        if not last_layer:
            for param in self.model.parameters():
                all_params.append(param)
        else:
            for param in self.model.last_layer.parameters():
                all_params.append(param)
    
        return all_params  # torch tensor of shape num_params
    
    def get_loss_on_idxs(self, data_idxs, is_test=False):
        bs = self.args.train_batch_size
        data_dl = self.prepare_loader(data_idxs, bs, drop_last=False,is_test=is_test, is_val=(not is_test))
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in data_dl:
                (X, y) = batch
                X,y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)  # shape: batch_size x 1
                if len(y.shape) != 1:
                    y = y.squeeze(dim=-1)
                if self.args.dataset.startswith("bio"):
                    loss = self.loss_fn(y_pred.squeeze(dim=-1), y).mean(dim=-1).detach().cpu().tolist()
                else:
                    loss = self.loss_fn(y_pred.squeeze(dim=-1), y).detach().cpu().tolist()
                total_loss.extend(loss)
                
        return total_loss

    def retrain(self, logger, train_idxs, val_idxs, n_epochs=256, batch_size=256, weight_decay=0.0, from_scratch=False, do_valid=False, **config):
        lr = self.args.lr
        train_dl = self.prepare_loader(train_idxs, batch_size, drop_last=True,is_test=False)
        valid_dl = None if not do_valid else self.prepare_loader(val_idxs, batch_size,drop_last=False,is_test=False, is_val=True)
        opt = self.get_opt(weight_decay)        
        if from_scratch:
            self.save_or_load_model(from_scratch=True, opt=opt,retrain=True)
            self.best_valid_mses = np.inf
            self.best_valid_acc = 0
        else:
            opt, _= self.save_or_load_model(from_scratch=False, opt=opt,retrain=True)
        offset = self.best_epoch
        lr_sched = config.get('lr_sched', 'lin')
        best_params = None
        for i in range(n_epochs):
            # do one training epoch
            self.model.train()
            if self.use_lr_sched:
                if not from_scratch:
                    epoch_lr = self.get_epoch_lr(i+offset,lr_sched,self.args.train_epochs) * lr
                else:
                    epoch_lr = self.get_epoch_lr(i,lr_sched,n_epochs) * lr
                for group in opt.param_groups:
                    group['lr'] = epoch_lr
            if self.args.dataset.startswith("cifar"):
                train_dl.dataset.val_mode = False
            for batch in train_dl:
                opt.zero_grad()
                (X, y) = batch
                X,y = X.to(self.device), y.to(self.device)
                y_pred = self.model(X)  # shape: batch_size x 1
                loss = self.loss_fn(y_pred.squeeze(dim=-1), y.squeeze(dim=-1)).mean()
                loss.backward()
                opt.step()
            if do_valid:
                if self.args.dataset.startswith("cifar"):
                    valid_dl.dataset.val_mode = True
                valid_mses,valid_acc = self.run_validation(valid_dl,len(val_idxs))
                if self.args.task_type == "regr":
                    if valid_mses.item() < self.best_valid_mses:
                        self.best_valid_mses = valid_mses.item()
                        self.best_epoch = i
                        best_params = copy.deepcopy(self.model.state_dict())
                else:
                    if valid_acc >= self.best_valid_acc:
                        self.best_valid_mses = valid_mses.item()#valid_acc
                        self.best_valid_acc = valid_acc
                        self.best_epoch = i
                        best_params = copy.deepcopy(self.model.state_dict())
                metric = 'Loss' if self.args.task_type == "regr" else 'Acc'
                value = valid_mses if self.args.task_type == "regr" else valid_acc
                logger.write(f"Epoch: {i} Valid {metric}: {value}")
                print(f"Epoch: {i} Valid {metric}: {value}")
        
        if do_valid:
            self.model.load_state_dict(best_params)
        if self.args.task_type == "clas":
            loss_idxs = self.test_model(logger, val_idxs, -1,if_retrain=True)['acc']
        else:
            loss_idxs = self.get_loss_on_idxs(val_idxs)
        opt, lr = self.save_or_load_model(from_scratch=False, opt=opt,retrain=True)
        return loss_idxs



