import numpy as np
from backbones import ModelTrainer, Timer
from utils import format_round_performance, init_strategy 
from baselines import UCBBatchAL, SCRiBLeSolver, ALBLExp4P
from itertools import islice
from autoal_engine import AutoALOrchestrator

def compute_strategy_fractions(logger, args, rd_num, strat_data_record, strat_list, mtrainer, train_idxs, val_idxs, test_idxs):
    '''
    Main function that computes the fractions for next AL round based on either influence or retraining style of SelectAL.
    
    KEY POINTS:
    1) We loosely call the performance change on deletion of points by SelectAL for any strategy as its influence and argmax it
    to obtain the next chosen strategy by SelectAL. Here exploration is set to 0.
    2) Veto is for obtaining the performance of pure strategies as in the manuscript. The strategy passed as veto is 
    assigned the whole budget. Exploration is 0 for veto cases
    '''

    #Equal fractions in the first round for all methods
    if rd_num == 0:
        for i, strategy in enumerate(strat_list):
            strat_data_record[strategy.name]["Frac"].append(1.0 / len(strat_list))
        return [1.0 / len(strat_list)] * len(strat_list), strat_data_record, []
    
    if args.always_equal: #For the baseline where we always use equal fractions for all strategies
        for i, strategy in enumerate(strat_list):
            strat_data_record[strategy.name]["Frac"].append(1.0 / len(strat_list))
            strat_data_record[strategy.name]["Influence"].append(0.0)
        return [1.0 / len(strat_list)] * len(strat_list), strat_data_record, []
    
    #Calculate influence scores (rewards) based on specified method. We obtain the time stabilized reward in case of FractAL.
    if args.inf_method == "indep_delete":
        from utils import indep_delete_influence_last_layer
        strat_data_record, rd_infl_record = indep_delete_influence_last_layer(logger, strat_data_record, strat_list, mtrainer, train_idxs, val_idxs, rd_num)
    elif args.inf_method == "selectAL":
        from utils import selectAL_influence
        strat_data_record, rd_infl_record = selectAL_influence(logger, strat_data_record, strat_list, mtrainer, val_idxs, rd_num)
        assert args.choose_best, "SelectAL chooses the one best strategy. Pass choose best!"
        assert args.expl_budget == 0.0, "SelectAL chooses the one best strategy. Pass exploration budget as 0!"
    elif args.inf_method == "retrain":
        from utils import retrain_influence
        strat_data_record, rd_infl_record = retrain_influence(logger, strat_data_record, strat_list, mtrainer, train_idxs, val_idxs, rd_num)
    elif args.inf_method == "corr":
        from utils import retrain_influence, indep_delete_influence_last_layer
        strat_data_record, rd_infl_record = indep_delete_influence_last_layer(logger, strat_data_record, strat_list, mtrainer, train_idxs, val_idxs, rd_num)
        strat_data_record, rd_infl_record = retrain_influence(logger, strat_data_record, strat_list, mtrainer, train_idxs, val_idxs, rd_num)
    
    #Calculate new fractions based on specified method
    if args.choose_best:
        ## Exploration budget to ensure p_min fraction to each strategy
        new_frac = [args.expl_budget/len(strat_list)]* len(strat_list)
        rem_budget = 1.0 - sum(new_frac)

        if args.veto == "none":
            # Since no veto, assign the remaining budget to the strategy with highest influence
            strat_influences = get_strategy_influence_list(strat_data_record, strat_list)
            chosen_strat = np.argmax(strat_influences)
        else:
            # Assign all budget to veto strategy
            assert rem_budget == 1.0, "Passed exploration for veto experiment. No exploration for veto!"
            chosen_strat = args.strategy.index(args.veto)
        new_frac[chosen_strat] += rem_budget
    else:
        old_frac_list = []
        rewards = []
        for strategy in strat_list:
            influence_scores = strat_data_record[strategy.name]["Influence"]
            rewards.append(influence_scores[-1])  # Storing time stabilized reward for each strategy
            old_frac_list.append(strat_data_record[strategy.name]["Frac"][-1]) # Storing the last fraction for each strategy
        if args.frac_method == "grad_descent":
            rewards = rewards / np.max(np.abs(rewards))
            new_frac = [max(old_f + delta, args.floor/args.al_batch_size) for old_f, delta in zip(old_frac_list, 0.15 * rewards)]
            new_frac = (np.array(new_frac) / np.sum(new_frac))
        elif args.frac_method == "mirror":
            scale_factor = args.adap_ss/np.max(np.abs(rewards)) #Adaptive mirror descent step size
            exp_fractions = np.multiply(np.array(old_frac_list), np.exp(scale_factor.item() * np.array(rewards))) # Mirror descent update
            new_frac = (exp_fractions / np.sum(exp_fractions)) #Normalize to obtain the fractions
            
        pmin = args.floor/args.al_batch_size
        new_frac = (1 - len(strat_list)*pmin) * new_frac + pmin #Enforcing p_min for each strategy
        new_frac = new_frac.tolist()

    # Record the new fractions for each strategy
    for i, strategy in enumerate(strat_list):
        strat_data_record[strategy.name]["Frac"].append(new_frac[i])
    
    return new_frac, strat_data_record, rd_infl_record

def get_strategy_influence_list(strat_data_record, strat):
    '''
    Returns the influence score of each strategy for the current round
    '''
    inf_list = []
    for strategy in strat:
        inf_list.append(strat_data_record[strategy.name]["Influence"][-1])
    return inf_list

def get_strategy_retrain_list(strat_data_record, strat):
    '''
    Returns the influence score of each strategy for the current round
    '''
    inf_list = []
    for strategy in strat:
        inf_list.append(strat_data_record[strategy.name]["Retrain"][-1])
    return inf_list

def frac_to_budget(frac_list, batch_size):
    '''
    Converts the allocation fraction to number of samples to be selected for each strategy 
    based on the AL batch size.
    '''
    budget_list = [int(frac * batch_size) for frac in frac_list]
    
    #Adjust strategy with highest fraction's budget to ensure total matches
    budget_list[0] += batch_size - sum(budget_list)
    return budget_list

def run_competition_active_learning(logger, args, dataset):
    # Initialize the strategies to be used in the experiment based on the input arguments
    str_func_list = init_strategy(args.strategy)
    strat = [str_func(args, dataset, lab_idx=[], unlab_idx=[]) for str_func in str_func_list]

    ### Initialize the bandit solver if the competition mode is bandit-based
    if args.comp_mode == "bandit_ucb":
        bandit_obj = UCBBatchAL(n=len(strat), c=args.bandit_c, pseudo_count=1e-6)
    elif args.comp_mode == "bandit_scrible":
        bandit_obj = SCRiBLeSolver(n=len(strat), eta=args.bandit_eta, kappa=args.kappa)
    elif args.comp_mode == "bandit_exp4p":
        bandit_obj = ALBLExp4P(n=len(strat), al_batch_size=args.al_batch_size)
    
    '''
    Initialize the data structure to store the strategy-wise acquired data, influence scores 
    and fractions across rounds for analysis and visualization purposes
    '''
    strat_data_record = {}
    for strategy in strat:
        strat_data_record[strategy.name] = {"lSet": [], "Influence": [0.0], "Frac": [], "Retrain": [0.0]} # Initialize influence and retrain scores to 0.0 for round 0
    
    ## Initialize the surrogate model trainer object which will be used to train the model
    mtrainer = ModelTrainer(data=dataset, args=args)

    #To store the formatted round-wise performance records to be returned at the end
    perf_records = []
    samp_proc_timer = Timer()
    rew_proc_timer = Timer()
    #Book keeping to avoid repeated extraction of indices
    val_idxs = dataset.valid_idxs.tolist()
    test_idxs = dataset.test_idxs.tolist()

    #Initialize AutoAL orchestrator if needed
    if args.comp_mode == "autoal":
        device = mtrainer.device
        autoal = AutoALOrchestrator(args, dataset, device, len(strat))
        new_batch = []

    for i in range(args.num_al_steps):
        logger.write(f"Active Learning Step {i}/{args.num_al_steps-1}")
        
        if i == 0:
            # Round 0 is to acquire the initial labeled set D_init
            lSet = dataset.init_train_idxs.tolist()
            uSet = dataset.pool_idxs.tolist()
        elif args.comp_mode == "autoal" and i > 1:
            #AutoAL: add the batch acquired in the previous round
            lSet.extend(new_batch)
            uSet = list(set(uSet) - set(new_batch))
        else:
            print(f"Round {i}: Strategy Fractions: {[strategy+':'+str(frac) for strategy, frac in zip(args.strategy, strat_fractions)]}")
            logger.write(f"Strategy Fractions for round {i}: {[strategy+':'+str(frac) for strategy, frac in zip(args.strategy, strat_fractions)]}")

            '''
            SAMPLING PROCEDURE FOR THE STRATEGIES BASED ON THEIR ALLOCATED FRACTIONS
            
            1) Sort the strategies based on their fractions and start with strategy with highest fraction
            2) Each strategy samples a whole batch of points from the same unlabeled pool independently 
            based on its own selection criteria
            3) To form the composite batch, we go down the sorted list and pick points from each strategy's sampled batch according to its
            allocated proportion/fraction. If a strategy's chosen point is already in the labeled set, we take the next point from the same strategy's sampled set. 
            We keep track of these repeats for analysis purposes.
            '''
            samp_proc_timer.start()
            sort_idx = np.argsort(strat_fractions)[::-1]
            all_sel_points = []
            for idx in sort_idx.tolist():
                if strat_fractions[idx] == 0.0:
                    str_lSet = []
                else:
                    str_lSet, _ = strat[idx].get_preds(logger, args.al_batch_size,i, trainer=mtrainer)
                all_sel_points.append(str_lSet)
            lSet = set()
            ## obtain the budget for each strategy based on the allocated fractions and the total AL batch size
            budget_list = frac_to_budget(np.array(strat_fractions)[sort_idx].tolist(), args.al_batch_size)
            ## Temporarily sort the strategies in descending order according to their fractions
            strat_order = [strat[idx].name for idx in sort_idx.tolist()]
            for idx, sel_points in enumerate(all_sel_points):
                print(f"Strategy {strat_order[idx]} allocated fraction: {strat_fractions[sort_idx[idx]]} budget: {budget_list[idx]} samples.")
                logger.write(f"Strategy {strat_order[idx]} allocated fraction: {strat_fractions[sort_idx[idx]]} budget: {budget_list[idx]} samples.")
                if budget_list[idx] == 0:
                    strat_data_record[strat_order[idx]]["lSet"].append([]) #Strategy wasn't allocated any budget
                    continue

                # Choose the first 'budget_list[idx]' points from sel_points that are not already in the labeled set
                str_sel_points = list(islice((elem for elem in sel_points if elem not in lSet), budget_list[idx]))

                # For record keeping, store the chosen point idxs in the strategy's records
                strat_data_record[strat_order[idx]]["lSet"].append(str_sel_points)
                
                # logger.write("Repeats for strategy {}: {}\n".format(strat_order[idx], repeats))
                lSet.update(str_sel_points)
            samp_proc_timer.pause()
            #--------- END OF SAMPLING PROCEDURE --------- We now have the composite batch
                
            uSet = list(set(strat[0].uSet) - lSet) # Obtain the updated uSet
            lSet = list(lSet) # Convert to list for further processing
            assert len(lSet) == args.al_batch_size, "Total selected samples do not match the AL batch size."
        
        #Update all strategies with the updated labeled and unlabeled sets
        for strategy in strat:
            strategy.update_lab_and_unlab_sets(lSet, uSet)
        logger.write(f"Updated labeled set size: {len(strat[0].lSet)}, unlabeled set size: {len(strat[0].uSet)}")    

        ## Training and Evaluation logic
        mtrainer.train(logger, strat[0].lSet, val_idxs, i) #strat[0].lSet is the total composite labeled set from beginning until current round

        #Obtain the performance metrics on train, val and test sets after training in the current round for record keeping and analysis purposes
        train_res, val_res, test_res = mtrainer.test_model(logger, strat[0].lSet, i), mtrainer.test_model(logger, val_idxs, i, if_val=True), mtrainer.test_model(logger, test_idxs,i, if_test=True)

        logger.write(f"After training in Round {i}: Train Loss: {train_res['loss']} \tVal Loss: {val_res['loss']} \tTest Loss: {test_res['loss']}")
        
        #Compute new strategy fractions based on influence scores
        
        if args.comp_mode == "autoal":
            #AutoAL acquisition (skip at the last round)
            # breakpoint()
            if i < args.num_al_steps - 1 and i > 0:

                rew_proc_timer.start()
                # Split current lSet into L_tr and L_va
                indices = np.random.permutation(len(lSet))
                split = int(np.floor(0.5 * len(lSet)))
                l_tr = [lSet[idx] for idx in indices[:split]]
                l_va = [lSet[idx] for idx in indices[split:]]

                #Compute strategies' votes on L_va
                p_votes = autoal.compute_p_votes(strat, l_tr, l_va, args.al_batch_size//2, logger, i, trainer=mtrainer)

                #Train AutoAL modules (GateHead, LossNet, trunk+clf)
                autoal.init_fresh()
                autoal.train_autoal(l_tr, l_va, p_votes, logger)
                rew_proc_timer.pause()

                #Compute strategies' votes on uSet for acquisition
                samp_proc_timer.start()
                p_votes_u = autoal.compute_p_votes(strat, lSet, uSet, args.al_batch_size, logger, i, trainer=mtrainer)
                
                #Acquire next batch from unlabeled pool
                new_batch = autoal.acquire(uSet, p_votes_u, args.al_batch_size)
                samp_proc_timer.pause()

            strat_fractions = [1.0 / len(strat)] * len(strat)
            rd_infl_record = []
            #Record fractions for each strategy (equal fractions for autoal)
            for j, strategy in enumerate(strat):
                strat_data_record[strategy.name]["Frac"].append(strat_fractions[j])
        elif args.comp_mode.startswith("bandit"):
            rew_proc_timer.start()
            if args.comp_mode == "bandit_ucb":
                if args.task_type == "clas":
                    bandit_obj.update_values(-1 if i == 0 else np.argmax(strat_fractions), val_res['acc']) # argmax of fraction in current round represents the action chosen for the current round
                elif args.task_type == "regr":
                    bandit_obj.update_values(-1 if i == 0 else np.argmax(strat_fractions), -val_res['loss'])
                chosen_strat_idx = bandit_obj.select_strategy(i)
                # Allocate all the budget to the chosen strategy
                strat_fractions = [0.0]*len(strat)
                strat_fractions[chosen_strat_idx] = 1.0
            elif args.comp_mode == "bandit_scrible":
                bandit_obj.update(cost=val_res['loss'], round_num=i) #Using val loss as the cost since scrible minimizes cost
                strat_fractions = bandit_obj.suggest_action().tolist() #Returns a point in the simplex, used directly as proportions for next round
            elif args.comp_mode == "bandit_exp4p":
                if args.task_type == "clas":
                    bandit_obj.update(score=val_res['acc'], round_num=i) #Using accuracy on val set as the reward
                elif args.task_type == "regr":
                    bandit_obj.update(score=1/(1+val_res['loss']), round_num=i) #Transformed val loss to obtain reward in [0,1]
                chosen_strat_idx = bandit_obj.suggest_action()
                # Allocate all the budget to the chosen strategy
                strat_fractions = [0.0]*len(strat)
                strat_fractions[chosen_strat_idx] = 1.0
            
            #Record the fractions for each strategy for analysis purposes
            for j, strategy in enumerate(strat):
                    strat_data_record[strategy.name]["Frac"].append(strat_fractions[j])
            #No influence here, so passing an empty list for influence record
            rd_infl_record = []
        else:
            # For non-bandit based competition modes, FractAL and SelectAL type competition
            rew_proc_timer.start()
            strat_fractions, strat_data_record, rd_infl_record = compute_strategy_fractions(logger, args, i, strat_data_record, strat, mtrainer, strat[0].lSet, val_idxs, test_idxs)
        rew_proc_timer.pause()
        '''
        Format the recorded data for the current round for analysis and visualization purposes at the end of the experiment
        Note that the fractions here are for the next round while the performance results are for the current round
        '''
        perf_records.append(format_round_performance(i, train_res, val_res, test_res, frac=strat_fractions, infl=get_strategy_influence_list(strat_data_record, strat), infl_rec=rd_infl_record, retrain=get_strategy_retrain_list(strat_data_record, strat)))
        logger.write("************* Finished Round {} *************".format(i))
    
    logger.write(f"Total time taken for sampling in all rounds: {samp_proc_timer.acc_time_total} seconds")
    logger.write(f"Total time taken for reward calculation in all rounds: {rew_proc_timer.acc_time_total} seconds")
    return perf_records