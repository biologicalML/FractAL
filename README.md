## Dependencies
There are 2 .yml files in the code.

Use environment.yml for running the experiments and obtaining the results
Use plot.yml for plotting purposes.

You can create environments using the command:
```
conda create --file <file_name>
```

environment.yml will create the environment by the name fractal. Activate the environment by:
```
conda activate fractal
```

Similarly, plot.yml will create the environment by the name plot. Activate the environment by:
```
conda activate plot
```

## Data Specifics

**For CIFAR 10:**

Download the train embeddings from _https://github.com/fbickfordsmith/ssl-embeddings_ and place them at the path `dataset/data/cifar10/cifar10_simclr_train.npy`. Please make sure the embeddings are L2 normalized.


**For BMDM (Macrophage data):**

The count data will be made public when the paper is published. 

The `process_data.py` script in the folder `dataset/data/bio_macrophage/` is used to process the data and obtain the X.npy, y.npy and other supporting files.

Similarly, the processing script for CD4+ T cell is also present.

**For Standard Regression Datasets:**

The X.npy and y.npy files are present in the respective folders to be used by the code.


## Running the experiments

We provide a specific example to run the setup C1 in our manuscript in the run_expt.sh for all the baselines and our method.

The arguments can be correspondingly changed for the regression and bio setups according to hyperparameter choices detailed in the paper.

## Plotting the results

We provide two files _plot_strat_perf.py_ and _plot_budget_allocation.py_ for plotting the different figures in the paper.

The scripts require you to manually add the folder names that you wish to plot. Further instructions are detailed in each script. Each script could be run by the command:
```
python <script_name> --dataset <dataset_name> --al_batch_size <AL Batch Size> --task_type <clas or regr> --legend_only 
```
where `legend_only` is to make sure the legend is plotted separately from the main plot.

## Allocation Diagnostic Metric

We provide the alloc_diagnostics.py file that can help compute the Best-Set and Worst-Set Overlap metric reported in the manuscript. By default, it computes Best-Set overlap. To compute worst, simply pass the `--worst` flag in the command. Example command:
```
#For Best-Set Overlap
python alloc_diagnostics.py

#For Worst-Set Overlap
python alloc_diagnostics.py --worst
```

Note that you are required to add the result folder names for all the pure strategy and adaptive methods across all the datasets to compute the full tables.