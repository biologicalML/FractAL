import os
import json
import numpy as np
import torch.serialization
from dataset import Task, TaskSplit, Data, DataSplit
from utils import log, get_res_directory, dump_expt_args, set_seeds
from al_main import run_competition_active_learning
from args import read_args

def sanity_checks(args):
    
    if args.task_type == "clas":
        try:
            assert args.emb in ["model", "simclr"]
        except:
            args.emb = "simclr"
    elif args.task_type == "regr":
        try:
            assert args.emb == "input"
        except:
            args.emb = "input"
    else:
        raise NotImplementedError(f"unrecognized task type '{args.task_type}'")
    return args

if __name__ == "__main__":
    torch.serialization.add_safe_globals([
        (np._core.multiarray.scalar, 'numpy.core.multiarray.scalar'),
        np.dtype,
        np.dtypes.Float64DType
    ])
    args = read_args()
    args = sanity_checks(args) #Sanity checks for the embeddings to be used in the experiment
    print(args)
    
    try:
        if args.dataset.startswith("cifar"):
            data_obj = Data(args)
            train_data, train_len = data_obj.getDataset(isTrain=True, isDownload=True)
            test_data, test_len = data_obj.getDataset(isTrain=False, isDownload=True)
            ds = DataSplit(args, train_data, test_data)
        else:
            task = Task.get_tabular_tasks(args)
            ds = TaskSplit(task, use_pool_for_normalization=args.use_pool_for_normalization) 
        print("Created dataset")
    except:
        raise NotImplementedError(f"unrecognized dataset '{args.dataset}'")

    #Create Results folder and update args
    args = get_res_directory(args)
    
    #Dump the al config contents to a json file
    dump_expt_args(args)

    human = log.HumanLogger(args.res_dir)
    expt_results = {}
    
    with human.for_id(f"trace_{args.dataset}") as hlog:
        for dsplit in range(args.n_data_splits):
            hlog.write(f"************* Starting data split {dsplit+1}/{args.n_data_splits} *************\n")
            print(f"************* Starting data split {dsplit+1}/{args.n_data_splits} *************\n")
            dsplit_res = {}
            '''
            Set the seeds for the data split and process the data (randomize the initial split of the
            data, and keep same for all runs for this data split)
            '''
            set_seeds(21*dsplit)
            if args.dataset.startswith("cifar"):
                ds.process_data(21*dsplit)
            else:
                ds.process_data(task, 21*dsplit)
            
            for run in range(args.n_runs):
                args.id = run
                set_seeds(args.id) #Set the seeds for the run (model init randomization)
                #Call to the main AL loop, returns the performance records for the run
                perf_records = run_competition_active_learning(hlog, args, ds)
                dsplit_res[f'run_{run+1}'] = perf_records
                print(f"Completed run {run+1}/{args.n_runs} for dataset {args.dataset} with strategy {args.strategy}")
                print("****************************************************************")
                hlog.write(f"************* Completed run {run+1}/{args.n_runs} *************\n")
            
            expt_results[f'dsplit_{dsplit+1}'] = dsplit_res

        #Dump the performance records to a json file
        with open(os.path.join(args.res_dir, f'performance_records.json'), 'w') as f:
            json.dump(expt_results, f, indent=4)