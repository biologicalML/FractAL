import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from dataset import TaskSplit
from typing import Optional, Callable, List
from torch import Tensor


class SimpleGateHead(nn.Module):
    def __init__(self, emb_dim, n_strategies):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_strategies)
        )

    def forward(self, x):
        return self.net(x)


class SimpleLossNet(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def loss_pred_loss(input, target, margin=1.0):
    assert len(input) % 2 == 0, 'batch size must be even'
    B = len(input) // 2
    diff_pred = input[:B] - input[B:].flip(0)
    diff_true = (target[:B] - target[B:].flip(0)).detach()
    sign = 2 * torch.sign(torch.clamp(diff_true, min=0)) - 1
    loss = torch.sum(torch.clamp(margin - sign * diff_pred, min=0))
    return loss / B


def compute_ratio_penalty(n_active, target_count):
    diff = abs(n_active - target_count)
    return (1 / (1 + math.exp(-0.5 * diff)) - 0.5) * 2


class ResNetTrunk(nn.Module):
    """ResNet with named submodules (matching torchvision conventions)
    so that load_pretrained works.  Exposes .model (trunk → embedding)
    and .last_layer (classifier head)."""

    def __init__(
        self,
        block,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_dropout: bool = False
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use_dropout = use_dropout
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.last_layer = nn.Linear(512 * block.expansion, num_classes)
        self.drop = nn.Dropout(p=0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        from backbones.resnet import conv1x1
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.use_dropout:
            x = self.drop(x)
        return x

    @property
    def model(self):
        return self


class AutoALOrchestrator:
    def __init__(self, args, dataset, device, n_strategies):
        self.args = args
        self.dataset = dataset
        self.device = device
        self.n_strategies = n_strategies
        self.trunk = None
        self.classifier = None
        self.gate_head = None
        self.loss_module = None
        self.is_cifar = args.dataset.startswith("cifar")

    def _get_embed_dim(self):
        from backbones.dataset_model_dict import dm_dict
        from dataset import TaskSplit
        rec = dm_dict[self.args.dataset]
        if rec["type"] == "mlp":
            hidden = rec["kwargs"].get("hidden_sizes", [])
            if hidden:
                return hidden[-1]
            if isinstance(self.dataset, TaskSplit):
                n_features = self.dataset.data.tensors['X'].shape[1]
            return n_features
        elif rec["type"] == "resnet18":
            return 512
        raise NotImplementedError

    def _get_backbone(self):
        from backbones.models import MLP
        from backbones.resnet import BasicBlock, model_urls, load_state_dict_from_url
        from backbones.dataset_model_dict import dm_dict
        rec = dm_dict[self.args.dataset]
        if rec["type"] == "mlp":
            if isinstance(self.dataset, TaskSplit):
                n_features = self.dataset.data.tensors['X'].shape[1]
            return MLP(n_features=n_features, act=self.args.model_actvn, **rec["kwargs"])
        elif rec["type"] == "resnet18":
            kwargs = rec["kwargs"].copy()
            num_classes = kwargs.pop("num_classes", 10)
            layers = kwargs.pop("layers", [2, 2, 2, 2])
            model = ResNetTrunk(BasicBlock, layers, num_classes=num_classes,
                                use_dropout=self.args.dropout)
            if self.args.autoal_pretrained:
                sd = load_state_dict_from_url(model_urls['resnet18'], progress=True)
                model_state = model.state_dict()
                sd = {k: v for k, v in sd.items()
                      if not k.startswith('fc.')
                      and k in model_state
                      and v.shape == model_state[k].shape}
                model.load_state_dict(sd, strict=False)
                print("Loaded pretrained ResNet18 weights for AutoAL trunk")
            return model
        raise NotImplementedError

    def _trunk_embed(self, x):
        embed = self.trunk(x)
        if self.is_cifar:
            embed = embed.view(embed.size(0), -1)
        return embed

    def init_fresh(self):
        model = self._get_backbone().to(self.device)
        self.trunk = model.model
        self.classifier = model.last_layer
        emb_dim = self._get_embed_dim()
        self.gate_head = SimpleGateHead(emb_dim, self.n_strategies).to(self.device)
        self.loss_module = SimpleLossNet(emb_dim).to(self.device)

    def compute_p_votes(self, strategies, l_tr, pool_idxs, al_batch_size, logger, round_num, **kwargs):
        """Run each strategy on pool_idxs → binary p(x) matrix.

        Temporarily swaps each strategy's lSet → l_tr and uSet → pool_idxs
        so that strategies have the correct labeled set for internal state
        (graph, clusters) but select exclusively from pool_idxs.

        Returns: (len(pool_idxs), K) binary numpy array.
        """
        n_pool = len(pool_idxs)
        p_votes = np.zeros((n_pool, self.n_strategies), dtype=np.float32)
        pool_set = set(pool_idxs)
        pos_map = {idx: pos for pos, idx in enumerate(pool_idxs)}

        for k, strategy in enumerate(strategies):
            orig_lSet = strategy.lSet
            orig_uSet = strategy.uSet
            strategy.lSet = l_tr
            strategy.uSet = pool_idxs
            picks, _ = strategy.get_preds(logger, al_batch_size, round_num, **kwargs)
            strategy.lSet = orig_lSet
            strategy.uSet = orig_uSet
            for p in picks:
                if p in pool_set:
                    p_votes[pos_map[p], k] = 1.0
        return p_votes

    def _make_loader_with_indices(self, idxs, batch_size, shuffle=True):
        if self.is_cifar:
            data = self.dataset.data
            dataset_with_idx = _IndexedDataset(data, idxs)
            return DataLoader(
                dataset_with_idx, batch_size=batch_size,
                sampler=SubsetRandomSampler(range(len(idxs))) if shuffle else None,
                shuffle=False, pin_memory=True, num_workers=2
            )
        else:
            ds = self.dataset if not isinstance(self.dataset, TaskSplit) else self.dataset.data
            X = ds.tensors['X'][idxs].cpu()
            y = ds.tensors['y'][idxs].cpu().squeeze()
            if y.dim() == 0:
                y = y.unsqueeze(0)
            positions = torch.arange(len(idxs))
            tensor_ds = TensorDataset(X, y, positions)
            return DataLoader(tensor_ds, batch_size=batch_size,
                              shuffle=shuffle, pin_memory=True)

    def train_autoal(self, l_tr, l_va, p_votes, logger):
        batch_size = self.args.train_batch_size//2
        tr_loader = self._make_loader_with_indices(l_tr, batch_size, shuffle=True)
        va_loader = self._make_loader_with_indices(l_va, batch_size, shuffle=True)
        va_iter = iter(va_loader)

        for tr_batch in tr_loader:
            try:
                va_batch = next(va_iter)
            except StopIteration:
                va_iter = iter(va_loader)
                va_batch = next(va_iter)

            x_tr, y_tr = tr_batch[0].to(self.device), tr_batch[1].to(self.device)
            x_va, y_va, va_pos = va_batch
            x_va = x_va.to(self.device)
            y_va = y_va.to(self.device)
            va_pos = va_pos.long()

            # ─── train_1 ───
            self.trunk.train()
            self.classifier.train()
            opt_t1 = torch.optim.Adam(
                list(self.trunk.parameters()) + list(self.classifier.parameters()),
                lr=self.args.lr_autoal_trunk, betas=(0.9, 0.999),
                weight_decay=self.args.autoal_wd
            )
            for niter in range(self.args.train_1_iters):
                embed = self._trunk_embed(x_tr)
                logits = self.classifier(embed)
                if self.args.task_type == "clas":
                    loss = F.cross_entropy(logits, y_tr)
                else:
                    loss = nn.MSELoss()(logits.squeeze(), y_tr.float())
                if niter%20 == 0:
                    print(f"WARMSTART: step {niter}, loss {loss.item():.4f}")
                opt_t1.zero_grad()
                loss.backward()
                opt_t1.step()

            # ─── train_2 ───
            opt_gate = torch.optim.SGD(
                self.gate_head.parameters(),
                lr=self.args.lr_autoal_gate, momentum=0.9,
                weight_decay=self.args.autoal_wd
            )
            opt_trunk2 = torch.optim.Adam(
                list(self.trunk.parameters()) + list(self.classifier.parameters()),
                lr=self.args.lr_autoal_trunk, betas=(0.9, 0.999),
                weight_decay=self.args.autoal_wd
            )
            opt_loss = torch.optim.SGD(
                self.loss_module.parameters(),
                lr=self.args.lr_autoal_module, momentum=0.9,
                weight_decay=self.args.autoal_wd
            )

            total_len = len(l_tr) + len(l_va)
            target_active = int(total_len * self.args.autoal_ratio)
            init_loss = None

            for step in range(self.args.autoal_steps):
                phase = step % 3

                embed = self._trunk_embed(x_va)
                logits = self.classifier(embed)
                if self.args.task_type == "clas":
                    ce = F.cross_entropy(logits, y_va, reduction='none')
                else:
                    ce = nn.MSELoss(reduction='none')(logits.squeeze(), y_va.float())
                ce_mean = ce.detach().mean()
                if init_loss is None:
                    init_loss = ce_mean

                batch_p = torch.from_numpy(p_votes[va_pos.cpu().numpy()]).to(self.device).float()

                if phase == 0:
                    # Trunk + classifier: score-weighted CE on L_va
                    gate_raw = self.gate_head(embed.detach())
                    score = (torch.sigmoid(gate_raw) * batch_p).sum(dim=1, keepdim=True)
                    opt_trunk2.zero_grad()
                    opt_gate.zero_grad()
                    opt_loss.zero_grad()
                    n = (score > 0.5).sum().item()
                    add = compute_ratio_penalty(n, target_active)
                    # breakpoint()
                    loss = (score.detach() * ce).mean() + 0.5 * add 
                    # print(f"Phase 0: step {step}, loss {loss.item():.4f}")
                    loss.backward()
                    opt_trunk2.step()

                elif phase == 1:
                    # GateHead: maximise agreement with strongest disagreement
                    ce_sg = ce.detach()
                    gate_raw = self.gate_head(self._trunk_embed(x_va).detach())
                    score = (torch.sigmoid(gate_raw) * batch_p).sum(dim=1,keepdim=True)
                    opt_gate.zero_grad()
                    opt_trunk2.zero_grad()
                    opt_loss.zero_grad()
                    n = (score > 0.5).sum().item()
                    add = compute_ratio_penalty(n, target_active)
                    loss_1 = (ce_sg * score).mean()
                    loss = -loss_1 - 2 * add
                    # print(f"Phase 1: step {step}, loss {loss.item():.4f}")
                    loss.backward()
                    opt_gate.step()

                else:
                    # LossNet: predict loss from trunk embedding
                    opt_loss.zero_grad()
                    opt_gate.zero_grad()
                    opt_trunk2.zero_grad()
                    B = len(x_va)
                    if B % 2 != 0:
                        x_even = x_va[:-1]
                        y_even = y_va[:-1]
                    else:
                        x_even = x_va
                        y_even = y_va
                    embed_even = self._trunk_embed(x_even)
                    pred_loss = self.loss_module(embed_even)
                    if self.args.task_type == "clas":
                        true_loss = F.cross_entropy(self.classifier(embed_even),
                                                    y_even, reduction='none').detach()
                    else:
                        true_loss = nn.MSELoss(reduction='none')(self.classifier(embed_even).squeeze(),
                                                                 y_even.float()).detach()
                    
                    if self.args.dataset.startswith("bio"):
                        true_loss = true_loss.view(true_loss.size(0), -1).mean(dim=1)
                    lpl = loss_pred_loss(pred_loss, true_loss)
                    lpl = torch.clamp(lpl, min=0, max=10)
                    # print(f"Phase 2: step {step}, loss {lpl.item():.4f}")
                    lpl.backward()
                    opt_loss.step()

    def acquire(self, uSet, p_votes, al_batch_size):
        self.trunk.eval()
        self.gate_head.eval()

        loader = self._make_loader_with_indices(uSet, self.args.train_batch_size,
                                                 shuffle=False)
        all_scores = []
        with torch.no_grad():
            for batch in loader:
                x, _, pos = batch
                x = x.to(self.device)
                pos = pos.long()
                embed = self._trunk_embed(x)
                gate_raw = self.gate_head(embed)
                gate_act = torch.sigmoid(gate_raw)
                batch_p = torch.from_numpy(p_votes[pos.cpu().numpy()]).to(self.device).float()
                scores = (gate_act * batch_p).sum(dim=1).cpu().numpy()
                all_scores.extend(scores.tolist())

        if not all_scores:
            return []

        top_idx = np.argsort(all_scores)[-al_batch_size:]
        return [uSet[i] for i in top_idx]


class _IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, idxs):
        self.base = base_dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        x, y = self.base[self.idxs[i]]
        return x, y, i
