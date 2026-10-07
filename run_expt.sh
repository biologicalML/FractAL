#!/bin/bash

# Pure method baselines
python main.py --dataset 'cifar10' --strategy 'probcover' 'coreset' 'random' --init_label_pct 500 --use_lr_sched --model_actvn 'relu' --val_pct 1024 --opt_name 'sgd' --lr 0.025 --train_epochs 50 --train_batch_size 100 --wd 0.0003 --model 'resnet18' --dropout --task_type 'clas' --al_batch_size 500 --probcover_delta 0.6 --expl_budget 0.0 --ewma 0.3 --emb 'simclr' --choose_best --veto 'random' --damping 0.01
python main.py --dataset 'cifar10' --strategy 'probcover' 'coreset' 'random' --init_label_pct 500 --use_lr_sched --model_actvn 'relu' --val_pct 1024 --opt_name 'sgd' --lr 0.025 --train_epochs 50 --train_batch_size 100 --wd 0.0003 --model 'resnet18' --dropout --task_type 'clas' --al_batch_size 500 --probcover_delta 0.6 --expl_budget 0.0 --ewma 0.3 --emb 'simclr' --choose_best --veto 'probcover' --damping 0.01
python main.py --dataset 'cifar10' --strategy 'probcover' 'coreset' 'random' --init_label_pct 500 --use_lr_sched --model_actvn 'relu' --val_pct 1024 --opt_name 'sgd' --lr 0.025 --train_epochs 50 --train_batch_size 100 --wd 0.0003 --model 'resnet18' --dropout --task_type 'clas' --al_batch_size 500 --probcover_delta 0.6 --expl_budget 0.0 --ewma 0.3 --emb 'simclr' --choose_best --veto 'coreset' --damping 0.01

#always equal baseline
python main.py --dataset "cifar10" --strategy "probcover" "coreset" "random" --init_label_pct 500 --use_lr_sched --model_actvn "relu" --val_pct 1024 --opt_name "sgd" --lr 0.025 --train_epochs 50 --train_batch_size 100 --wd 0.0003 --model "resnet18" --dropout --task_type "clas" --al_batch_size 500 --probcover_delta 0.6 --expl_budget 0.0 --ewma 1.0 --always_equal --emb "simclr"

#SelectAL baseline
python main.py --dataset "cifar10" --strategy "probcover" "coreset" "random" --init_label_pct 500 --use_lr_sched --model_actvn "relu" --val_pct 1024 --opt_name "sgd" --lr 0.025 --train_epochs 50 --train_batch_size 100 --wd 0.0003 --model "resnet18" --dropout --task_type "clas" --al_batch_size 500 --probcover_delta 0.6 --choose_best --expl_budget 0.0 --ewma 1.0 --inf_method "selectAL" --retr_scratch --emb "simclr"

#Our Method
python main.py --dataset 'cifar10' --strategy 'probcover' 'coreset' 'random' --init_label_pct 500 --use_lr_sched --model_actvn 'relu' --val_pct 1024 --opt_name 'sgd' --lr 0.025 --train_epochs 50 --train_batch_size 100 --wd 0.0003 --model 'resnet18' --dropout --task_type 'clas' --al_batch_size 500 --probcover_delta 0.6 --expl_budget 0.006 --ewma 0.5 --emb 'simclr' --adap_ss 3.0 --damping 0.01