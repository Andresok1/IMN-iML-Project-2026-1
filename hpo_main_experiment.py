import argparse
import json
import os
import time

import numpy as np
import optuna
import pandas as pd

from main_experiment import main    
from search_spaces import hpo_space_imn, hpo_space_tabresnet        
from utils import get_dataset
from tools import update_summary, global_json_calculation, generate_cluster_feature_plots, save_test_train_data


def objective(
    trial: optuna.trial.Trial,
    args: argparse.Namespace,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    categorical_indicator: np.ndarray,
    attribute_names: np.ndarray,
    dataset_name: str,
    output_directory: str,
    cluster_len: float, 
) -> float:
    """The objective function for hyperparameter optimization.

    Args:
        trial: The optuna trial object.
        args: The arguments for the experiment.
        X_train: The training examples.
        y_train: The training labels.
        X_valid: The validation examples.
        y_valid: The validation labels.
        categorical_indicator: The categorical indicator for the features.
        attribute_names: The feature names.
        dataset_name: The name of the dataset.

    Returns:
        The test AUROC.

    """

    if args.interpretable:
        hp_config = hpo_space_imn(trial)    
    else:
        hp_config = hpo_space_tabresnet(trial)  

    output_info = main(
        args,
        hp_config,
        X_train,
        y_train,
        X_valid,
        y_valid,
        categorical_indicator,
        attribute_names,
        dataset_name,
        output_directory,
        cluster_len,
    )

    return output_info['test_auroc']


def hpo_main(args):
    ###################
    #IML: This function was modified to include more parformance metrics, control the output_info saving, and cluster handling.
    ###################
    '''Manage clusters, save output info and summary update, handle output directories and hpo tuning
    Args:
        args: The arguments for the experiment.
    Returns:   
        summary_path: The path to the summary file.
        attribute_names: The feature names.
    '''
    """The main function for hyperparameter optimization."""

    info_cluster, attribute_names, categorical_indicator = get_dataset(
        args.dataset_id,
        test_split_size=args.test_split_size,
        seed=args.seed,
        encode_categorical=True,
        hpo_tuning=args.hpo_tuning,
        create_clusters=args.create_clusters,
        visualization= args.visualization,
        cluster_type = args.cluster_type
    )

    timestamp_exists= False

    for cluster_id, info in info_cluster.items():

        if cluster_id == 'cluster_whole':
            dataset_name = "cluster_whole"  
        else:
            dataset_name = f"cluster_{cluster_id}"

        print(f"****{dataset_name}*****")


        X_train = info['X_train']
        X_test = info['X_test']

        y_train = info['y_train']
        y_test = info['y_test']

        cluster_len = info['cluster_len']

        if args.hpo_tuning:
            X_valid = info['X_valid']
            y_valid = info['y_valid']
        else:
            X_valid, y_valid = None, None
        
        positive_train = len(y_train[y_train == 1])
        positive_test = len(y_test[y_test == 1])
        if args.hpo_tuning:
            positive_valid = len(y_valid[y_valid == 1])

        negative_train = len(y_train[y_train == 0])
        negative_test = len(y_test[y_test == 0])
        if args.hpo_tuning:
            negative_valid = len(y_valid[y_valid == 0])

        model_name = 'inn' if args.interpretable else 'tabresnet'

        if not timestamp_exists:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()) 
            timestamp_exists = True
            timestamp_saved = timestamp     

        if args.create_clusters:
            output_directory = os.path.join(
                args.output_dir,
                model_name,
                f'dataset_{args.dataset_id}_{timestamp_saved}',
                f'seed_{args.seed}', 
                f'cluster_{cluster_id}'
            )
        else:
            output_directory = os.path.join(
                args.output_dir,
                model_name,
                f'dataset_{args.dataset_id}_no_clusters_{timestamp_saved}',
                f'seed_{args.seed}', 
                f'cluster_{cluster_id}'
            )

        os.makedirs(output_directory, exist_ok=True)

        best_params = None
        if args.hpo_tuning:

            time_limit = 60 * 60
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=args.seed),
            )

            # queue default configurations as the first trials
            if args.interpretable:
                study.enqueue_trial(
                    {
                        'nr_epochs':150,
                        'batch_size': 64,
                        'learning_rate': 0.01,
                        'weight_decay': 0.01,
                        'weight_norm': 0.1,
                        'dropout_rate': 0.25,
                    }
                )
            else:
                study.enqueue_trial(
                    {
                        'nr_epochs': 150,
                        'batch_size': 64,
                        'learning_rate': 0.01,
                        'weight_decay': 0.01,
                        'dropout_rate': 0.25,
                    }
                )

            try:
                study.optimize(
                    lambda trial: objective(
                        trial,
                        args,
                        X_train,
                        y_train,
                        X_valid,
                        y_valid,
                        categorical_indicator,
                        attribute_names,
                        dataset_name,
                        output_directory,
                        cluster_len,
                    ),
                    n_trials=args.n_trials,
                    timeout=time_limit,
                )
            except optuna.exceptions.OptunaError as e:
                print(f'Optimization stopped: {e}')

            best_params = study.best_params
            trial_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
            trial_df.to_csv(os.path.join(output_directory, 'trials.csv'), index=False)

        X_train = pd.concat([X_train, X_valid], axis=0)
        y_train = np.concatenate([y_train, y_valid], axis=0)

        output_info = main(
            args,
            best_params if args.hpo_tuning else None,
            X_train,
            y_train,
            X_test,
            y_test,
            categorical_indicator,
            attribute_names,
            dataset_name,
            output_directory,
            cluster_len,
        )

        output_info['positive_train'] = positive_train
        output_info['positive_test'] = positive_test
        output_info['positive_valid'] = positive_valid if args.hpo_tuning else None

        output_info['negative_train'] = negative_train
        output_info['negative_test'] = negative_test   
        output_info['negative_valid'] = negative_valid if args.hpo_tuning else None

        parent_dir = os.path.dirname(output_directory)
        summary_path = os.path.join(parent_dir, 'summary.json')
        
        update_summary(summary_path, output_info)

        with open(os.path.join(output_directory, 'output_info.json'), 'w') as f:
            json.dump(output_info, f)

    save_test_train_data(info_cluster, parent_dir, attribute_names)

    return summary_path, attribute_names


            
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--nr_blocks",
        type=int,
        default=2,
        help="Number of levels in the hypernetwork",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Number of hidden units in the hypernetwork",
    )
    parser.add_argument(
        "--augmentation_probability",
        type=float,
        default=0,
        help="Probability of data augmentation",
    )
    parser.add_argument(
        "--scheduler_t_mult",
        type=int,
        default=2,
        help="Multiplier for the scheduler",
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed',
    )
    parser.add_argument(
        '--dataset_id',
        type=int,
        default=31,
        help='Dataset id',
    )
    parser.add_argument(
        '--test_split_size',
        type=float,
        default=0.2,
        help='Test size',
    )
    parser.add_argument(
        '--nr_restarts',
        type=int,
        default=3,
        help='Number of learning rate restarts',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save the results',
    )
    parser.add_argument(
        '--interpretable',
        action='store_true',
        default=False,
        help='Whether to use interpretable models',
    )
    parser.add_argument(
        '--encoding_type',
        type=str,
        default='ordinal',
        help='Encoding type',
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='classification',
        help='If we are doing classification or regression.',
    )
    parser.add_argument(
        '--hpo_tuning',
        action='store_true',
        help='Whether to perform hyperparameter tuning',
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=100,
        help='Number of trials for hyperparameter tuning',
    )
    parser.add_argument(
        '--disable_wandb',
        action='store_true',
        help='Whether to disable wandb logging',
    )
    parser.add_argument(
        '--create_clusters',
        action='store_true',
        help='Whether to create clusters in the dataset',
    )
    parser.add_argument(
        '--visualization',
        action='store_true',
        help='Whether to create clusters in the dataset',
    )
    parser.add_argument(
        '--cluster_type',
        type=int,
        default=1,
        help='select cluster method (1=Gower, 2=Kmeans)',
    )

    args = parser.parse_args()

    summary_path, feature_list = hpo_main(args)
    
    if args.create_clusters:
        global_json_calculation(summary_path, feature_list)

    generate_cluster_feature_plots(summary_path)

    print(f"All data saved in:{summary_path}")
