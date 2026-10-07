import os
import json

def get_res_directory(args, plot=False):
    args.res_dir = f'./results/{args.dataset}/{str(args.strategy)}_{args.model}_{args.emb}{'_pt' if args.pretrained else ''}_{args.opt_name}_{args.init_label_pct}dinit_'
    args.res_dir += f'{args.val_pct}dval_'
    args.res_dir += f'{args.n_data_splits}dsplits_{args.n_runs}runs_{args.al_batch_size}bs_{args.num_al_steps}steps'
    args.res_dir += f'{"_" + args.model_actvn if args.model_actvn != "relu" else ""}{f"_{args.wd}wd" if args.wd > 0.0 else ""}'
    args.res_dir += f'{"_retrsc" if args.retr_scratch else "_retrbest"}'
    args.res_dir += f"_{args.lr}lr"
    if args.always_equal:
        args.res_dir += "_eqfrac"
    else:
        if args.choose_best:
            args.res_dir += "_cb"
            args.res_dir += f"_{args.veto}" if args.veto != "none" else ""
        
    args.res_dir += f'_{args.ewma}ewma' if args.ewma < 1.0 else ''
    args.res_dir += f'_{args.inf_method[:5]}'
    args.res_dir += "_noexpl" if args.expl_budget == 0.0 else f"expl{args.expl_budget}"
    args.res_dir += f'_{args.frac_method[:5]}' if not args.always_equal and not args.choose_best else ""
    args.res_dir += f'{f"_{args.comp_mode}_{args.bandit_c}" if args.comp_mode == "bandit_ucb" else ""}'
    args.res_dir += f'{f"_{args.comp_mode}_{args.bandit_eta}" if args.comp_mode == "bandit_scrible" else ""}'
    args.res_dir += f'{f"_{args.comp_mode}" if args.comp_mode == "bandit_exp4p" else ""}'
    args.res_dir += f'_adapss{args.adap_ss}' if args.adap_ss != 1.0 else ''
    args.res_dir += f'_damp{args.damping}' if args.damping != 0.5 else ''

    if not plot:
        os.makedirs(args.res_dir, exist_ok=True)
    return args

def dump_expt_args(args):
    with open(os.path.join(args.res_dir, 'al_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

def format_round_performance(rd_num, train_res, val_res, test_res, frac=None, infl=None, infl_rec=None, **kwargs):
    if frac is None and infl is None and infl_rec is None:
        return {
            "round": rd_num,
            "train_perf": train_res,
            "val_perf": val_res,
            "test_perf": test_res,
        }
    else:
        return {
            "round": rd_num,
            "train_perf": train_res,
            "val_perf": val_res,
            "test_perf": test_res,
            "fractions": frac,
            "influences": infl,
            "infl_rec": infl_rec,
            "retrain": kwargs.get("retrain", [])
        }

def init_strategy(strat_list):
    strat_func_list = []
    for strat_name in strat_list:
        strat_name = strat_name.lower()
        if strat_name == "random":
            from strategies import Random
            strat_func_list.append(Random)
        elif strat_name == "coreset":
            from strategies import CoreSet
            strat_func_list.append(CoreSet)
        elif strat_name == "typiclust":
            from strategies import TypiClust
            strat_func_list.append(TypiClust)
        elif strat_name == "probcover":
            from strategies import ProbCover
            strat_func_list.append(ProbCover)
        elif strat_name == "minmargin":
            from strategies import MinMargin
            strat_func_list.append(MinMargin)
        elif strat_name == "leastconf":
            from strategies import LeastConf
            strat_func_list.append(LeastConf)
        elif strat_name == "maxent":
            from strategies import MaxEnt
            strat_func_list.append(MaxEnt)
        elif strat_name == "badge":
            from strategies import BADGE
            strat_func_list.append(BADGE)
        else:
            raise NotImplementedError(f"Strategy '{strat_name}' not implemented.")
    return strat_func_list

def set_seeds(seed, benchmark=False):
    import random
    import torch
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = benchmark