import argparse
import json
import os

import numpy as np
import optuna
import pandas as pd

from main_experiment import main
from search_spaces import hpo_space_imn, hpo_space_tabresnet
from utils import get_dataset, get_dataset_from_csv, get_next_run_id, plot_top_features, plot_clusters_pca
from sklearn.metrics import (
    roc_auc_score, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score
)

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

    tmp_dir = os.path.join(args.output_dir, "hpo_temp")
    os.makedirs(tmp_dir, exist_ok=True)

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
        output_dir=tmp_dir,
    )

    return output_info['test_auroc']

BATCH_SIZE = 32 # Standard war 64
# ValueError: '16' not in (32, 64, 128, 256, 512)

def hpo_main(args):
    """The main function for hyperparameter optimization."""

    # Angepasst fürs einlesen von einer lokalen CSV
    if args.dataset_path is not None:
        info = get_dataset_from_csv(
            csv_path=args.dataset_path,
            target_column=args.target_column,
            test_split_size=args.test_split_size,
            seed=args.seed,
            encode_categorical=True,
            hpo_tuning=args.hpo_tuning,
            n_clusters = 2
        )
        dataset_name = os.path.splitext(
            os.path.basename(args.dataset_path)
        )[0]
    else:
        info = get_dataset(
            args.dataset_id,
            test_split_size=args.test_split_size,
            seed=args.seed,
            encode_categorical=True,
            hpo_tuning=args.hpo_tuning,
        )
        dataset_name = info["dataset_name"]

    #Ausgabe neue Ordnerstruktur
    dataset_out_dir = os.path.join(args.output_dir, dataset_name)
    run_id = get_next_run_id(dataset_out_dir)
    run_out_dir = os.path.join(dataset_out_dir, run_id)
    os.makedirs(run_out_dir, exist_ok=True)

    # Clustering
    if info.get("clustered", False):
        print(f"Training {info['n_clusters']} cluster-specific models")
        print(f"Output directory: {run_out_dir}")

        plot_clusters_pca(
            info["X_for_clustering"],
            info["cluster_labels"],
            out_dir=run_out_dir,
            filename="clusters_pca.png",
        )

        for cid, cluster_info in info["clusters"].items():
            print(f"\n===== Cluster {cid} =====")

            cluster_out_dir = os.path.join(run_out_dir, f"cluster_{cid}")
            os.makedirs(cluster_out_dir, exist_ok=True)

            attribute_names = cluster_info["attribute_names"]
            categorical_indicator = cluster_info["categorical_indicator"]

            X_train = cluster_info["X_train"]
            y_train = cluster_info["y_train"]
            X_test = cluster_info["X_test"]
            y_test = cluster_info["y_test"]

            if args.hpo_tuning:
                X_valid = cluster_info["X_valid"]
                y_valid = cluster_info["y_valid"]

            # Cluster Informationen extra speichern in cluster_info
            n_train = len(X_train)
            n_test = len(X_test)
            n_valid = len(X_valid) if args.hpo_tuning else 0
            n_total = n_train + n_test + n_valid

            # Labelverteilung
            train_label_dist = pd.Series(y_train).value_counts().to_dict()
            test_label_dist = pd.Series(y_test).value_counts().to_dict()
            valid_label_dist = pd.Series(y_valid).value_counts().to_dict() if args.hpo_tuning else {}

            cluster_stats = {
                "cluster_id": cid,
                "n_total": int(n_total),
                "n_train": int(n_train),
                "n_valid": int(n_valid),
                "n_test": int(n_test),
                "label_distribution_train": train_label_dist,
                "label_distribution_valid": valid_label_dist,
                "label_distribution_test": test_label_dist,
            }

            with open(os.path.join(cluster_out_dir, "cluster_info.json"), "w") as f:
                json.dump(cluster_stats, f, indent=2)

            # Cluster Train Daten abspeichern
            train_df = X_train.copy()
            train_df[args.target_column] = y_train
            train_df.to_csv(
                os.path.join(cluster_out_dir, "train_data.csv"),
                index=False
            )

            # Cluster Test Daten abspeichern
            test_df = X_test.copy()
            test_df[args.target_column] = y_test
            test_df.to_csv(
                os.path.join(cluster_out_dir, "test_data.csv"),
                index=False
            )

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
                            'nr_epochs': 500,
                            'batch_size': BATCH_SIZE,
                            'learning_rate': 0.01,
                            'weight_decay': 0.01,
                            'weight_norm': 0.1,
                            'dropout_rate': 0.25,
                        }
                    )
                else:
                    study.enqueue_trial(
                        {
                            'nr_epochs': 500,
                            'batch_size': BATCH_SIZE,
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
                        ),
                        n_trials=args.n_trials,
                        timeout=time_limit,
                    )
                except optuna.exceptions.OptunaError as e:
                    print(f'Optimization stopped: {e}')

                best_params = study.best_params

                trial_df = study.trials_dataframe(
                    attrs=('number', 'value', 'params', 'state')
                )
                trial_df.to_csv(
                    os.path.join(cluster_out_dir, 'trials.csv'),
                    index=False,
                )

                X_train = pd.concat([X_train, X_valid], axis=0)
                y_train = np.concatenate([y_train, y_valid], axis=0)

            output_info = main( #Hier erstelle ich das Modell
                args,
                best_params,
                X_train,
                y_train,
                X_test,
                y_test,
                categorical_indicator,
                attribute_names,
                dataset_name,
                cluster_out_dir, # Model.pt richtig im Ordner speichern
            )

            # Plot erzeugen
            plot_top_features(output_info, cluster_out_dir)

            with open(os.path.join(cluster_out_dir, 'output_info.json'), 'w') as f:
                json.dump(output_info, f, indent=2)

        # Cluster vergleichbar machen mit einem einzelnen Modell
        rows = []

        for cid in info["clusters"].keys():
            cdir = os.path.join(run_out_dir, f"cluster_{cid}")

            out_path = os.path.join(cdir, "output_info.json")
            info_path = os.path.join(cdir, "cluster_info.json")

            if not (os.path.exists(out_path) and os.path.exists(info_path)):
                print(f"Skipping cluster {cid}: missing output_info.json or cluster_info.json")
                continue

            with open(out_path, "r") as f:
                out = json.load(f)
            with open(info_path, "r") as f:
                cinfo = json.load(f)

            n_test = int(cinfo.get("n_test", 0))

            rows.append({
                "cluster_id": int(cid),
                "n_total": int(cinfo.get("n_total", 0)),
                "n_train": int(cinfo.get("n_train", 0)),
                "n_valid": int(cinfo.get("n_valid", 0)),
                "n_test": n_test,
                "test_accuracy": out.get("test_accuracy", None),
                "test_auroc": out.get("test_auroc", None),
                "train_time": out.get("train_time", None),
                "inference_time": out.get("inference_time", None),
                "test_balanced_accuracy": out.get("test_balanced_accuracy", None),
                "test_precision": out.get("test_precision", None),
                "test_recall": out.get("test_recall", None),
                "test_f1": out.get("test_f1", None),
                "top_features": out.get("top_features", None),
                "top_features_weights": out.get("top_features_weights", None),
            })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(run_out_dir, "cluster_overview.csv"), index=False)

        combined = {
            "n_clusters": int(info.get("n_clusters", len(df))),
            "total_test_samples": int(df["n_test"].sum()) if len(df) > 0 else 0,
            "weighted_test_accuracy": output_info.get("test_accuracy"),
            "weighted_test_auroc": output_info.get("test_auroc"),
            "weighted_train_time": output_info.get("train_time", None),
            "weighted_inference_time": output_info.get("inference_time", None),
            "weighted_test_balanced_accuracy": output_info.get("test_balanced_accuracy"),
            "weighted_test_precision": output_info.get("test_precision"),
            "weighted_test_recall": output_info.get("test_recall"),
            "weighted_test_f1": output_info.get("test_f1"),
        }

        def weighted_mean(col: str) -> float:
            vals = pd.to_numeric(df[col], errors="coerce")
            weights = df["n_test"].astype(float)
            mask = vals.notna() & (weights > 0)
            if mask.sum() == 0:
                return float("nan")
            return float((vals[mask] * weights[mask]).sum() / weights[mask].sum())

        for m in [
            "test_accuracy",
            "test_auroc",
            "train_time",
            "inference_time",
            "test_balanced_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
        ]:
            if m in df.columns:
                combined[f"weighted_{m}"] = weighted_mean(m)

        # Speichern anpassen für globale Metriken
        def compute_global_metrics_from_predictions(pred_df: pd.DataFrame) -> dict:
            y_true = pred_df["y_true"].astype(int).to_numpy()
            y_pred = pred_df["y_pred"].astype(int).to_numpy()
            y_prob = pred_df["y_prob"].astype(float).to_numpy()

            out = {
                "total_test_samples": int(len(pred_df)),
                "test_accuracy": float(accuracy_score(y_true, y_pred)),
                "test_balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "test_precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "test_recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }

            # AUROC braucht beide Klassen
            if len(np.unique(y_true)) == 2:
                out["test_auroc"] = float(roc_auc_score(y_true, y_prob))
            else:
                out["test_auroc"] = None

            return out

        # Feature weights kombinieren
        if args.interpretable:
            combined_feature_weights = {}
            total_weight = 0.0

            for r in rows:
                n_test = float(r.get("n_test", 0))
                feats = r.get("top_features")
                ws = r.get("top_features_weights")

                if not feats or not ws or n_test <= 0:
                    continue

                total_weight += n_test

                for f, w in zip(feats, ws):
                    combined_feature_weights[f] = combined_feature_weights.get(f, 0.0) + float(w) * n_test

            # normalisieren
            if total_weight > 0 and len(combined_feature_weights) > 0:
                for f in combined_feature_weights:
                    combined_feature_weights[f] /= total_weight

                # sortieren nach Größe
                sorted_items = sorted(
                    combined_feature_weights.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )

                combined["combined_top_features"] = [k for k, _ in sorted_items]
                combined["combined_top_features_weights"] = [v for _, v in sorted_items]
            else:
                combined["combined_top_features"] = []
                combined["combined_top_features_weights"] = []

        with open(os.path.join(run_out_dir, "combined_metrics.json"), "w") as f:
            json.dump(combined, f, indent=2)

        print("\n=== Combined (weighted) metrics ===")
        for k, v in combined.items():
            print(f"{k}: {v}")

        # Globale Metriken
        all_preds = []

        for cid in info["clusters"].keys():
            cdir = os.path.join(run_out_dir, f"cluster_{cid}")
            pred_path = os.path.join(cdir, "predictions.csv")
            if not os.path.exists(pred_path):
                print(f"Skipping cluster {cid}: predictions.csv fehlt")
                continue

            dfp = pd.read_csv(pred_path)
            dfp["cluster_id"] = int(cid)  # für Analyse
            all_preds.append(dfp)

        if len(all_preds) > 0:
            global_pred_df = pd.concat(all_preds, axis=0, ignore_index=True)

            # predictions kombiniert abgespeichert
            global_pred_df.to_csv(os.path.join(run_out_dir, "global_predictions.csv"), index=False)

            global_metrics = compute_global_metrics_from_predictions(global_pred_df)
            global_metrics["n_clusters"] = int(info.get("n_clusters", len(all_preds)))

            with open(os.path.join(run_out_dir, "global_metrics.json"), "w") as f:
                json.dump(global_metrics, f, indent=2)

            #print("\n=== Global metrics (aus global_predictions.csv) ===")
            #print(json.dumps(global_metrics, indent=2))
        else:
            print("Keine predictions.csv gefunden, keine globalen Metriken berechenbar.")

        return

    # Alter Code, kein Clustering
    attribute_names = info['attribute_names']
    categorical_indicator = info['categorical_indicator']

    X_train = info['X_train']
    X_test = info['X_test']

    y_train = info['y_train']
    y_test = info['y_test']

    if args.hpo_tuning:
        X_valid = info['X_valid']
        y_valid = info['y_valid']

    model_name = 'inn' if args.interpretable else 'tabresnet'
    output_directory = os.path.join(
        args.output_dir,
        model_name,
        f'{args.dataset_id}',
        f'{args.seed}',
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
                    'nr_epochs': 500,
                    'batch_size': BATCH_SIZE,
                    'learning_rate': 0.01,
                    'weight_decay': 0.01,
                    'weight_norm': 0.1,
                    'dropout_rate': 0.25,
                }
            )
        else:
            study.enqueue_trial(
                {
                    'nr_epochs': 500,
                    'batch_size': BATCH_SIZE,
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
                ),
                n_trials=args.n_trials,
                timeout=time_limit,
            )
        except optuna.exceptions.OptunaError as e:
            print(f'Optimization stopped: {e}')

        best_params = study.best_params
        trial_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        trial_df.to_csv(os.path.join(run_out_dir, 'trials.csv'), index=False)

    # concatenate train and validation
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
        run_out_dir
    )

    combined = {
        "n_clusters": 1,
        "total_test_samples": int(len(X_test)),
        "weighted_test_accuracy": output_info.get("test_accuracy", None),
        "weighted_test_auroc": output_info.get("test_auroc", None),
        "weighted_train_time": output_info.get("train_time", None),
        "weighted_inference_time": output_info.get("inference_time", None),
        "weighted_test_balanced_accuracy": output_info.get("test_balanced_accuracy"),
        "weighted_test_precision": output_info.get("test_precision"),
        "weighted_test_recall": output_info.get("test_recall"),
        "weighted_test_f1": output_info.get("test_f1"),
    }

    with open(os.path.join(run_out_dir, "combined_metrics.json"), "w") as f:
        json.dump(combined, f, indent=2)

    # Plot erzeugen
    plot_top_features(output_info, run_out_dir)

    with open(os.path.join(run_out_dir, 'output_info.json'), 'w') as f:
        json.dump(output_info, f)


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

    # Neues Argument --dataset_path für lokale CSV
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Path to a local CSV dataset',
    )

    # Neues Argument --dataset_path für lokale CSV
    parser.add_argument(
        '--target_column',
        type=str,
        default=None,
        help='Target column name for CSV datasets',
    )

    args = parser.parse_args()

    hpo_main(args)
