import argparse

def read_args():
    parser = argparse.ArgumentParser()
    ## GENERAL AL ARGS
    parser.add_argument(
        "--n_runs",
        type=int,
        default=10,
        help="Number of runs for each split.",
    )
    parser.add_argument(
        "--n_data_splits",
        type=int,
        default=5,
        help="Number of data splits for each setup.",
    )
    parser.add_argument(
        "--id",
        type=int,
        default=21,
        help="Random seed.",
    )
    parser.add_argument(
        "--al_batch_size",
        type=int,
        default=256,
        help="Batch size for active learning.",
    )
    parser.add_argument(
        "--num_al_steps",
        type=int,
        default=10,
        help="Number of active learning iteration.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        nargs="*",
        default=["random"],
        help="Active learning strategy(ies) to use.",
    )
    ###----------------------------------------------------
    ## DATA RELATED ARGS
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset",
        help="Directory to the data.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="diamonds",
        help="Name of the dataset.",
    )
    parser.add_argument(
        "--test_percent",
        type=float,
        default=0.2,
        help="Percentage of data to be used as test set. Can also pass an integer number for a fixed size test set",
    )
    parser.add_argument(
        "--use_pool_for_normalization",
        action="store_true",
        help="Whether to use the pool data for normalization. Only for standard regression datasets",
    )
    parser.add_argument(
        "--init_label_pct",
        type=float,
        default=0.2,
        help="Initial percentage of labeled data. Can also pass an integer for fixed size",
    )
    parser.add_argument(
        "--val_pct",
        type=float,
        default=0.1,
        help="Percentage of pool to be used as val set. Can also be an integer",
    )
    parser.add_argument(
        "--aug_method",
        type=str,
        default="hflip",
        help="Augmentation method for CIFAR 10 data",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="regr",
        help="Type of task: Regression or classification",
    )
    parser.add_argument(
        "--emb",
        type=str,
        default="simclr",
        choices=["model", "input", "simclr"],
        help="What embedding to use for the data",
    )
    ###----------------------------------------------------
    ## SURROGATE MODEL RELATED ARGS    
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        help="Surrogate Model to use.",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.01,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--use_lr_sched",
        action="store_true",
        help="Whether to use a scheduler for learning rate while training the surrogate",
    )
    parser.add_argument(
        "--model_actvn",
        type=str,
        default="relu",
        help="Activation function to use in the model.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning Rate for the surrogate model",
    )
    parser.add_argument(
        "--opt_name",
        type=str,
        default="adam",
        help="Optimizer to use in the model.",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=256,
        help="Number of epochs to train the model for",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=256,
        help="Batch size for train and val loader",
    )
    parser.add_argument(
        "--dropout",
        action="store_true",
        help="Whether to use the dropout or not",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Whether to use the pretrained weights for init or not",
    )
    parser.add_argument(
        "--topk_mse",
        type=int,
        nargs="*",
        default=[20],
        help="List of Top k genes to consider for MSE calculation",
    )
    ###----------------------------------------------------
    ## BANDIT ARGS
    parser.add_argument(
        "--comp_mode",
        type=str,
        default="infl",
        choices=["bandit_ucb", "bandit_scrible", "bandit_exp4p", "infl", "autoal"],
        help="Whether to use the bandit competition modes or influence-style competition mode",
    )
    parser.add_argument(
        "--bandit_c",
        type=float,
        default=1.0,
        help="Exploration parameter for bandit strategies for UCB",
    )
    parser.add_argument(
        "--bandit_eta",
        type=float,
        default=0.08,
        help="Learning Rate for Scrible Bandit strategy",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.0,
        help="Kappa parameter for the Scrible bandit baseline",
    )
    ###----------------------------------------------------
    ## AUTOAL ARGS
    parser.add_argument(
        "--autoal_ratio",
        type=float,
        default=0.3,
        help="Target gate activity ratio for AutoAL",
    )
    parser.add_argument(
        "--autoal_steps",
        type=int,
        default=400,
        help="Number of train_2 (3-phase) steps per batch-pair",
    )
    parser.add_argument(
        "--train_1_iters",
        type=int,
        default=200,
        help="Number of train_1 (trunk+clf only) iterations per batch-pair",
    )
    parser.add_argument(
        "--lr_autoal_trunk",
        type=float,
        default=0.01,
        help="Learning rate for trunk+classifier during AutoAL training",
    )
    parser.add_argument(
        "--lr_autoal_gate",
        type=float,
        default=0.01,
        help="Learning rate for GateHead during AutoAL training",
    )
    parser.add_argument(
        "--lr_autoal_module",
        type=float,
        default=0.01,
        help="Learning rate for LossNet during AutoAL training",
    )
    parser.add_argument(
        "--autoal_wd",
        type=float,
        default=5e-4,
        help="Weight decay for AutoAL training",
    )
    parser.add_argument(
        "--autoal_pretrained",
        action="store_true",
        help="Whether to use pretrained weights for AutoAL ResNet trunk",
    )
    ###----------------------------------------------------
    ## INFLUENCE ARGS
    parser.add_argument(
        "--inf_method",
        type=str,
        default="indep_delete",
        choices=["indep_delete", "selectAL", "retrain", "corr"],
        help="Whether using FractAL style influence or SelectAL style.",
    )
    parser.add_argument(
        "--retr_scratch",
        action="store_true",
        help="Whether to retrain from the best model or from scratch",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.5,
        help="Damping for IHVP stabilisation",
    )
    parser.add_argument(
        "--ewma",
        type=float,
        default=1.0,
        help="Parameter to control EWMA averaging of influence rewards",
    )
    ###----------------------------------------------------
    ## FRACTION DECISION ARGS
    parser.add_argument(
        "--choose_best",
        action="store_true",
        help="Whether to choose the best in competition or do fractions",
    )
    parser.add_argument(
        "--frac_method",
        type=str,
        default="mirror",
        choices=["mirror", "grad_descent"],
        help="Which method to use for fraction selection in the competition",
    )
    parser.add_argument(
        "--expl_budget",
        type=float,
        default=0.16,
        help="Fraction of AL batch size to be used as exploration always, equally among the other strategies",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=1.0,
        help="Minimum fraction of AL batch size to be used for each strategy in the competition",
    )
    parser.add_argument(
        "--adap_ss",
        type=float,
        default=1.0,
        help="Adaptive step size for mirror descent",
    )
    parser.add_argument(
        "--veto",
        type=str,
        default="none",
        choices=["random", "badge", "coreset", "leastconf", "maxent", "minmargin", "probcover", "typiclust", "none"],
        help="Which strategy to veto as the best always - Pure Strategy Baselines",
    )
    parser.add_argument(
        "--always_equal",
        action="store_true",
        help="Whether to always use equal fractions for all strategies",
    )
    ###----------------------------------------------------
    ## STRATEGY SPECIFIC ARGS
    parser.add_argument(
        "--probcover_delta",
        type=float,
        default=0.6,
        help="Delta value for probcover strategy.",
    )
    ###----------------------------------------------------
    ## PLOTTING ARGS
    parser.add_argument(
        "--legend_outside",
        action="store_true",
        help="Place the legend outside the plot on the right",
    )
    parser.add_argument(
        "--legend_only",
        action="store_true",
        help="Save the legend as a separate PDF and omit it from the main plot",
    )
    ###----------------------------------------------------

    args = parser.parse_args()
    return args