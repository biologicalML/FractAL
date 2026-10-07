from .hess_cg import get_inverse_hvp_cg
import torch
import numpy as np
from .custom_utils import init_strategy
import timeit

def get_ewma_value(infl_list, alpha, rd_num):
    val = alpha * infl_list[-1]
    if rd_num > 1:
        val += (1-alpha)*infl_list[-2]
    return val

def indep_delete_influence_last_layer(logger, strat_data_record, strat_list, mtrainer, train_idxs, val_idxs, rd_num):
    """Compute influence scores for each strategy based on independent deletion.

    Args:
        strat_data_record: Dictionary recording data for each strategy.
        strat_list: List of strategy instances.
        mtrainer: ModelTrainer instance.
        train_idxs: List of training indices.
        val_idxs: List of validation indices.
        rd_num: Current round number
    Returns:
        Updated strat_data_record with influence scores.
    """
    if mtrainer.args.choose_best and mtrainer.args.veto != "none":
        #Since we are to veto a strategy, avoid calculating influence scores
        for strategy in strat_list:
            strat_data_record[strategy.name]["Influence"].append(0.0)
        return strat_data_record, []
    
    #Get the gradient of the loss on validation set
    start_t = timeit.default_timer()
    grad = mtrainer.get_gradients(val_idxs, last_layer=True) #list of length num_param_groups
    end_t = timeit.default_timer()
    print(f'Total time taken for val gradients: {end_t - start_t} seconds')
        
    #Hessian isn't calculated explicitly, use CG method to find the IHVP
    verify_tol = 1 if mtrainer.args.task_type == "clas" else 1e-2
    start_t = timeit.default_timer()
    ihvp = get_inverse_hvp_cg(logger, mtrainer, grad, train_idxs, cg_iters=1000, damping=mtrainer.args.damping, tol=1e-8, verify_tol=verify_tol, last_layer=True) #List of length num_param_groups
    end_t = timeit.default_timer()
    print(f'Total time taken for IHVP: {end_t - start_t} seconds')

    for strategy in strat_list:
        added_indices = strat_data_record[strategy.name]["lSet"][-1]
        if len(added_indices) == 0:
            influence_score = 0.0
        else:
            ## Calculate the gradients for the training points chosen by the strategy
            start_t = timeit.default_timer()
            strat_train_grads = mtrainer.get_gradients(added_indices, last_layer=True) #shape is num_params x 1
            end_t = timeit.default_timer()
            print(f'Total time taken for train gradients: {end_t - start_t} seconds')
            influence_score = 0.0

            #Compute the inner product between ihvp and train grads to obtain the influence
            start_t = timeit.default_timer()
            flat_ihvp = torch.cat([elem.reshape(-1) for elem in ihvp])
            flat_train_grads = torch.cat([elem.reshape(-1) for elem in strat_train_grads])
            influence_score += torch.dot(flat_ihvp, flat_train_grads).item()
            end_t = timeit.default_timer()
            print(f'Total time taken for strategy influence: {end_t - start_t} seconds')

            #Divide by the size of train set for stability. For our method, we divide by the max reward which nullifies this factor anyways!
            influence_score = influence_score / len(train_idxs)
            influence_score = influence_score
        print(strategy.name + ':' + str(influence_score))
        logger.write(f"Influence score for strategy {strategy.name}: {influence_score}")
        smooth_influence_score = (mtrainer.args.ewma * influence_score) + (1 - mtrainer.args.ewma) * strat_data_record[strategy.name]["Influence"][-1]
        logger.write(f"Smooth Influence score for strategy {strategy.name}: {smooth_influence_score}")
        strat_data_record[strategy.name]["Influence"].append(smooth_influence_score)

    return strat_data_record, []

def get_strategy_idxs(logger, strategy, mtrainer, rd_num):
    train_idxs = strategy.lSet
    num_preds = max(int(0.05*len(train_idxs)),1) ## SelectAL deletes at least 1 point for every strategy to calculate its utility

    ## Initialize the same strategy object again with the same data but no labeled data and unlabeled pool as the current training data
    strat = init_strategy([strategy.name])[0](mtrainer.args, strategy.data, lab_idx=[], unlab_idx=train_idxs)
    ## Obtain the points this strategy would sample
    str_preds,_ = strat.get_preds(logger, num_preds, rd_num, trainer=mtrainer)
    return str_preds

def selectAL_influence(logger, strat_data_record, strat_list, mtrainer, val_idxs, rd_num):
    if mtrainer.args.task_type == "clas":
        perf_before = mtrainer.best_valid_acc
    else:
        val_idx_losses = mtrainer.get_loss_on_idxs(val_idxs)
        perf_before = np.mean(val_idx_losses).item()
    
    train_idxs = strat_list[0].lSet
    for strategy in strat_list:
        ## Use the strategy to find the subset of points it wishes to delete from the current train set
        str_preds = get_strategy_idxs(logger, strategy, mtrainer, rd_num)

        ## Obtain the performance after retraining the model on the updated train set
        perf_after = mtrainer.retrain(logger, list(set(train_idxs) - set(str_preds)), val_idxs, weight_decay=mtrainer.args.wd, from_scratch=True,n_epochs=mtrainer.args.train_epochs, batch_size=mtrainer.args.train_batch_size, do_valid=True)

        ## Calculate the change in performance depending on task type
        if mtrainer.args.task_type == "clas": # We expect to measure reduction in accuracy
            influence_score = perf_before - np.mean(perf_after).item()
        else:    # We expect the loss to increase
            influence_score = np.mean(perf_after).item() - perf_before
        print(strategy.name + ':' + str(influence_score))
        logger.write(f"Influence score for strategy {strategy.name}: {influence_score}")
        strat_data_record[strategy.name]["Influence"].append(influence_score)
        
    return strat_data_record, []

def retrain_influence(logger, strat_data_record, strat_list, mtrainer, train_idxs, val_idxs, rd_num):
    if mtrainer.args.task_type == "clas":
        top_loss_val_idxs = val_idxs
        val_loss_before = mtrainer.best_valid_acc
    else:
        val_idx_losses = mtrainer.get_loss_on_idxs(val_idxs)
        val_loss_before = np.mean(val_idx_losses).item()

    for strategy in strat_list:
        added_indices = strat_data_record[strategy.name]["lSet"][-1]
        loss_after = mtrainer.retrain(logger, list(set(train_idxs) - set(added_indices)), val_idxs, weight_decay=mtrainer.args.wd, from_scratch=True,n_epochs=mtrainer.args.train_epochs, batch_size=mtrainer.args.train_batch_size, do_valid=True)
        if mtrainer.args.task_type == "clas":
            influence_score = val_loss_before - np.mean(loss_after).item()
        else:    
            influence_score = np.mean(loss_after).item() - val_loss_before
        
        print(strategy.name + ':' + str(influence_score))
        logger.write(f"Influence score for strategy {strategy.name}: {influence_score}")
        if mtrainer.args.inf_method == "corr":
            smooth_influence_score = (mtrainer.args.ewma * influence_score) + (1 - mtrainer.args.ewma) * strat_data_record[strategy.name]["Retrain"][-1]
            strat_data_record[strategy.name]["Retrain"].append(smooth_influence_score)
        else:
            smooth_influence_score = (mtrainer.args.ewma * influence_score) + (1 - mtrainer.args.ewma) * strat_data_record[strategy.name]["Influence"][-1]
            strat_data_record[strategy.name]["Influence"].append(smooth_influence_score)
        
        logger.write(f"Smooth Influence score for strategy {strategy.name}: {smooth_influence_score}")
            
    return strat_data_record, []
