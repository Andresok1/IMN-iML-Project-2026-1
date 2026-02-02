# (iML-Project)Clustering Analysis of Datasets Using Mesomorphic Networks

Interpretable Machine Learning methods are often evaluated on global datasets, assuming homogeneous data distributions and uniform model behaviour. However, real world tabular datasets often exhibit structural heterogeneity, where different subpopulations follow distinct predictive patterns. In this project, we study how model performance and interpretability change when the data is partitioned into clusters prior to training. We propose a pipeline that performs clustering in the raw feature space using the Gower distance guided by the silhouette criterion and clustering k means. Model training and interpretation are carried out using the Interpretable Mesomorphic Network architecture previously proposed in the literature, which provides instance level feature importance scores to compare explanatory patterns across clusters. 

This work is based on the original repository developed by [Arlind Kadra, Sebastian Pineda Arango, and Josif Grabocka]. The codebase has been adapted and extended as part of an academic project focused on interpretable machine learning.


## Setting up the virtual environment

```
conda create -n imn python=3.10
conda activate imn
pip install -r requirements.txt
```

## Running the code

The entry script to run IMN and TabResNet is `hpo_main_experiment.py`. 

The main arguments for `hpo_main_experiment.py` are:

- `--nr_blocks`: Number of residual blocks in the hypernetwork.
- `--hidden_size`: The number of hidden units per-layer.
- `--augmentation_probability`: The probability with which data augmentation will be applied.
- `--scheduler_t_mult`: Number of restarts for the learning rate scheduler.
- `--seed`: The random seed to generate reproducible results.
- `--dataset_id`: The dataset to open (0=Churn dataset).
- `--test_split_size`: The fraction of total data that will correspond to the test set.
- `--nr_restarts`: Number of restarts for the learning rate scheduler.
- `--output_dir`: Directory where to store results.
- `--interpretable`: If interpretable results should be generated, basically if IMN should be used or the TabResNet architecture.
- `--mode`: Takes two arguments, `classification` and `regression`. 
- `--hpo_tuning`: Whether to enable hyperparameter optimization. 
- `--nr_trials`: The number of trials when performing hyperparameter optimization. 
- `--disable_wandb`: Whether to disable wandb logging. 
- `--create_clusters`: Whether to create clusters or just consider the whole dataset.
- `--visualization`: Whether to generate a visualization of your dataset/cluster set.
- `--cluster_type`: Whether to cluster via Gower or Kmeans. 


**A minimal example of running IMN**:

```
Experiment 1:
python hpo_main_experiment.py --hpo_tuning --n_trials 3 --seed 0 --disable_wandb --interpretable --dataset_id 0  --create_clusters --cluster_type 1

Experiment 2:
python eval_saved_model.py --model_dir C:\Users\Andres\Documents\Repos\iML\Project\IMN\inn\dataset_0_2026-01-31_15-19-07\seed_0\cluster_whole --target_column churn

```

## Plots

The plots that are included in our paper were generated from the functions in the module `tools.py`.
The plots expect the following result folder structure:

```
├── inn (results folder)
│   ├── dataset_id-date_timestamp
│   │   ├── seed
│   │   │   ├── cluster_NAME
│   │   │   │   ├── output_info.json
│   │   │   │   ├── top_features.png
│   │   │   │   ├── shap_summary.png
│   │   │   ├── cluster_global_ranking_mean.png

```

## Citation

```
@inproceedings{
kadra2024interpretable,
title={Interpretable Mesomorphic Networks for Tabular Data},
author={Arlind Kadra and Sebastian Pineda Arango and Josif Grabocka},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=PmLty7tODm}
}
```
