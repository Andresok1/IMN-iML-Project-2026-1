#(iML-Project )Interpretable Mesomorphic Networks for Tabular Data

Mesomorphic networks use Deep Neural Networks to construct a local linear intrinsic model for every instance. This approach allows for a highly interpretable predictions while retaining the strong predictive performance of neural networks. Interpretability is achieved both at the model level and the instance level. In this case, interpretability is not applied as a post-hoc method, but rather integrated directly into the prediction process. This combination allows to use the great performance of Neural Networks and at the same time provides interpretable results of linear models. The advantages of both approaches are preserved.

Through extensive experiments was demostrated that...

This work is based on the original repository developed by [Arlind Kadra, Sebastian Pineda Arango, and Josif Grabocka]. The codebase has been adapted and extended as part of an academic project focused on interpretable machine learning.


## Setting up the virtual environment

```
conda create -n imn python=3.10
conda activate imn
pip install -r requirements.txt
```

## Running the code

The entry script to run IMN and TabResNet is `hpo_main_experiment.py`. 
The entry script to run the baseline methods (CatBoost, Random Forest, Logistic Regression, Decision Tree and TabNet) is `hpo_baseline_experiment.py`.

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
python hpo_main_experiment.py --hpo_tuning --n_trials 3 --seed 0 --disable_wandb --interpretable --dataset_id 0  --create_clusters --cluster_type 1

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
