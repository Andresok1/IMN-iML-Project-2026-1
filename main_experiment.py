import argparse
import os
import time
from typing import Dict
from unittest import result

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, confusion_matrix, precision_score, recall_score

import torch
import wandb

from models.model import Classifier
from sklearn.metrics import balanced_accuracy_score 
from sklearn.metrics import f1_score
import shap
import pandas as pd
import matplotlib.pyplot as plt


def main(
    args: argparse.Namespace,
    hp_config: Dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    categorical_indicator: np.ndarray,
    attribute_names: np.ndarray,
    dataset_name: str,
    output_directory: str,
    cluster_len: float,
) -> Dict:
    ###################
    #IML: This function was modified to include more parformance metrics and SHAP explainability included
    ###################
    """Main entry point for the experiment.

    Args:
        args: The arguments for the experiment.
        hp_config: The hyperparameter configuration.
        X_train: The training examples.
        y_train: The training labels.
        X_test: The test examples.
        y_test: The test labels.
        categorical_indicator: The categorical indicator for the features.
        attribute_names: The feature names.
        dataset_name: The name of the dataset.
        output_directory: The output directory for the experiment.
        cluster_len: The length of the cluster (if applicable).

    Returns:
        output_info: A dictionary with the main results from the experiment.
    """
    dev = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_id = args.dataset_id

    if hp_config is None:
        hp_config = {
            'nr_epochs': 500,
            'batch_size': 64,
            'learning_rate': 0.01,
            'weight_decay': 0.01,
            'weight_norm': 0.1,
            'dropout_rate': 0.25,
        }

    seed = args.seed

    X_train = X_train.to_numpy()
    X_train = X_train.astype(np.float32)
    X_test = X_test.to_numpy()
    X_test = X_test.astype(np.float32)

    nr_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
    unique_classes, class_counts = np.unique(y_train, axis=0, return_counts=True)
    nr_classes = len(unique_classes)

    dataset_classes = {}
    for i in range(nr_classes):                      
        dataset_classes[i] = []     

    for index, label in enumerate(y_train):             
        dataset_classes[label].append(index)

    majority_class_nr = -1
    for i in range(nr_classes):                             
        if len(dataset_classes[i]) > majority_class_nr:
            majority_class_nr = len(dataset_classes[i])

    examples_train = []
    labels_train = []

    for i in range(nr_classes):                             
        nr_instances_class = len(dataset_classes[i])
        if nr_instances_class < majority_class_nr:
            
            oversampled_indices = np.random.choice(            
                dataset_classes[i],
                majority_class_nr - nr_instances_class,
                replace=True,
            )
            examples_train.extend(X_train[dataset_classes[i]])      
            
            labels_train.extend(
                y_train.iloc[dataset_classes[i]].tolist()
                if hasattr(y_train, "iloc")
                else y_train[dataset_classes[i]].tolist()
            )


            for index in oversampled_indices:                       
                examples_train.append(X_train[index])
                
                labels_train.append(
                    y_train.iloc[index] if hasattr(y_train, "iloc") else y_train[index]
                )
        else:
            examples_train.extend(X_train[dataset_classes[i]])      
            
            labels_train.extend(
                y_train.iloc[dataset_classes[i]].tolist()
                if hasattr(y_train, "iloc")
                else y_train[dataset_classes[i]].tolist()
            )

    network_configuration = {                                  
        'nr_features': nr_features,
        'nr_classes': nr_classes if nr_classes > 2 else 1,
        'nr_blocks': args.nr_blocks,
        'hidden_size': args.hidden_size,
        'dropout_rate': hp_config['dropout_rate'],
    }


    interpretable = args.interpretable
    model_name = 'inn' if interpretable else 'tabresnet'
    if not args.disable_wandb:                
        wandb.init(
            project='INN',
            config=args,
        )
        wandb.config['model_name'] = model_name
        wandb.config['dataset_name'] = dataset_name


    os.makedirs(output_directory, exist_ok=True)

    args.nr_epochs = hp_config['nr_epochs']
    args.learning_rate = hp_config['learning_rate']
    args.batch_size = hp_config['batch_size']
    args.weight_decay = hp_config['weight_decay']
    args.weight_norm = hp_config['weight_norm'] if 'weight_norm' in hp_config else 0.1
    args.dropout_rate = hp_config['dropout_rate']

    model = Classifier(                                             
        network_configuration,
        args=args,
        categorical_indicator=categorical_indicator,
        attribute_names=attribute_names,
        model_name=model_name,
        device=dev,
        output_directory=output_directory,
        disable_wandb=args.disable_wandb,
    )

    start_time = time.time()                                            
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    if interpretable:
        test_predictions, weight_importances = model.predict(X_test, y_test, return_weights=True)        

    else:
        test_predictions = model.predict(X_test, y_test)

    train_predictions = model.predict(X_train, y_train)

    inference_time = time.time() - start_time - train_time

    # SHAP
    def f(X):
        return model.predict(X)

    background = shap.sample(X_train, 25, args.seed)
    rng = np.random.default_rng(args.seed)
    if len(X_test) < 100:
        idx = rng.choice(len(X_test), size=100, replace=True)
    else: 
        idx = rng.choice(len(X_test), size=100, replace=False)
    X_shap = X_test[idx]

    explainer = shap.Explainer(f, background)
    shap_exp = explainer(X_shap)

    plt.figure()

    shap.dependence_plot("gender", shap_exp.values, X_shap, feature_names=attribute_names, interaction_index="auto", show=False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, "shap_dependence_gender.png"), dpi=200)
    plt.close()



    df_sv = pd.DataFrame(shap_exp.values, columns=attribute_names)
    df_sv.insert(0, "row_id", idx)  # original row indices aus X_test
    df_sv.to_csv(
        os.path.join(output_directory, "shap_values.csv"),
        index=False,
    )

    mean_abs = np.abs(shap_exp.values).mean(axis=0)
    df_imp = pd.DataFrame({"feature": attribute_names, "mean_abs_shap": mean_abs}) \
        .sort_values("mean_abs_shap", ascending=False)
    df_imp.to_csv(
        os.path.join(output_directory, "shap_importance.csv"),
        index=False,
    )

    shap.summary_plot(shap_exp.values, X_shap, feature_names=attribute_names, show=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_directory, "shap_summary.png"),
        dpi=200,
    )
    plt.close()

    # Bar Plot (global)
    plt.figure()
    plt.barh(df_imp["feature"].head(20)[::-1], df_imp["mean_abs_shap"].head(20)[::-1])
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_directory, "shap_bar_top20.png"),
        dpi=200,
    )
    plt.close()
    # End SHAP

    test_predictions = test_predictions.cpu().numpy()               
    train_predictions = train_predictions.cpu().numpy()

    if interpretable:
        weight_importances = weight_importances.cpu().detach().numpy()

  
    y_test = y_test.tolist()
    y_train = y_train.tolist()

    if args.mode == 'classification':                   
        test_auroc = roc_auc_score(
            y_test,
            test_predictions,
            multi_class='raise' if nr_classes > 2 else 'ovo',
        )
        train_auroc = roc_auc_score(
            y_train,
            train_predictions,
            multi_class='raise' if nr_classes > 2 else 'ovo',
        )

        # threshold the predictions if the model is binary
        if nr_classes == 2:                                             
            # threshold the predictions if the model is binary
            test_predictions = (test_predictions > 0.5).astype(int)    
            train_predictions = (train_predictions > 0.5).astype(int)
        else:
            test_predictions = np.argmax(test_predictions, axis=1)
            train_predictions = np.argmax(train_predictions, axis=1)

        test_accuracy = accuracy_score(y_test, test_predictions)        
        train_accuracy = accuracy_score(y_train, train_predictions)

        test_balance_accuracy = balanced_accuracy_score(y_test,test_predictions)
        train_balance_accuracy = balanced_accuracy_score(y_train,train_predictions)
        
        test_f1 = f1_score(y_test, test_predictions)

        test_precision = precision_score(y_test, test_predictions)
        test_recall = recall_score(y_test, test_predictions)


        if not args.disable_wandb:                                      
            wandb.run.summary["Test:accuracy"] = test_accuracy
            wandb.run.summary["Test:auroc"] = test_auroc
            wandb.run.summary["Train:accuracy"] = train_accuracy
            wandb.run.summary["Train:auroc"] = train_auroc
    else:                                                              
        test_mse = mean_squared_error(y_test, test_predictions)
        train_mse = mean_squared_error(y_train, train_predictions)
        if not args.disable_wandb:                                                           
            wandb.run.summary["Test:mse"] = test_mse
            wandb.run.summary["Train:mse"] = train_mse
    if args.mode == 'classification':
        output_info = {
            'dataset_name': dataset_name,
            'train_auroc': train_auroc,
            'train_accuracy': train_accuracy,
            'train_balance_accuracy':train_balance_accuracy,
            'test_auroc': test_auroc,
            'test_accuracy': test_accuracy,
            'test_balance_accuracy': test_balance_accuracy,
            'test_f1' : test_f1,
            'test_recall': test_recall,
            'test_precision' : test_precision,
            'train_time': train_time,
            'inference_time': inference_time,
            'cluster_len': cluster_len,
        }
    else:
        output_info = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_time': train_time,
            'inference_time': inference_time,
        }

    if interpretable:
        # remove the above line if you need the per example importance
        weight_importances = np.mean(weight_importances, axis=0)

        sorted_idx = np.argsort(weight_importances)[::-1]                   
        top_features = [attribute_names[i] for i in sorted_idx]

        output_info['top_features'] = top_features                         
        output_info['top_features_weights'] = weight_importances[sorted_idx].tolist()
        if not args.disable_wandb:
            wandb.run.summary["Top_features"] = top_features
            wandb.run.summary["Top_features_weights"] = weight_importances[sorted_idx]

    if not args.disable_wandb:
        wandb.finish()

    return output_info


