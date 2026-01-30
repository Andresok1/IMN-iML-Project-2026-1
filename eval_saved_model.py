import os
import json
import argparse
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from models.model import Classifier


def retest(args):

    # Load model
    model_path = os.path.join(args.model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)   

    model_dummy = args.model_dir
    tarjet_value = args.target_column
   # Dummy args für Classifier

    args = argparse.Namespace(
        mode= "classification",
        nr_epochs=0,
        batch_size=0,
        learning_rate=0.0,
        weight_decay=0.0,
        scheduler_t_mult=2,
        nr_restarts=1,
        weight_norm=0.0,
        augmentation_probability=0.0,
        dropout_rate=0.25,
        disable_wandb=True,
        nr_blocks= 2,
        hidden_size= 128,
        interpretable= True,
        model_dir= model_dummy,
        target_column= tarjet_value
    )

    model_name = "inn" if args.interpretable else "tabresnet"
    
    first_check = os.path.dirname(os.path.dirname(model_path))
    cluster_path_1 = os.path.join(first_check, "Cluster_1")
    test_file = os.path.join(cluster_path_1, "test_data_cluster_1.csv")
    df = pd.read_csv(test_file)
    X_test = df.drop(columns=["churn"]).to_numpy().astype(np.float32)


    # Für predict reicht das als Platzhalter
    categorical_indicator = [False] * X_test.shape[1]
    attribute_names = [f"f{i}" for i in range(X_test.shape[1])]

    # Netzwerk muss gleich aufgebaut werden wie beim Training
    network_configuration = {
        "nr_features": int(X_test.shape[1]),
        "nr_classes": 1,  # bei Telco => 1
        "nr_blocks": args.nr_blocks,
        "hidden_size": args.hidden_size,
        "dropout_rate": args.dropout_rate,
    }

    clf = Classifier(
        network_configuration=network_configuration,
        args=args,
        categorical_indicator=categorical_indicator,
        attribute_names=attribute_names,
        model_name=model_name,
        output_directory=args.model_dir,
        disable_wandb=True,
    )

    state = torch.load(model_path)

    clf.model.load_state_dict(state)
    clf.model.eval()

    clf.ensemble_snapshots = [state]

    experiment_folder = os.path.dirname(os.path.dirname(model_path))

    Summary_retest = []

    for cluster_id in os.listdir(experiment_folder): 
        cluster_path = os.path.join(experiment_folder, cluster_id)
        
        if not os.path.isdir(cluster_path):
            continue

        if cluster_id == "summary.json" or cluster_id == "summary_retest.json" :
            continue

        for fname in os.listdir(cluster_path):
            if fname.startswith("test_data_cluster_") and fname.endswith(".csv"):
                test_file = os.path.join(cluster_path, fname)

        df = pd.read_csv(test_file)

        if tarjet_value not in df.columns:
            raise KeyError(f"Target nicht in {test_file} gefunden.")

        y_test = df[args.target_column].to_numpy().astype(int)
        X_test = df.drop(columns=[args.target_column]).to_numpy().astype(np.float32)

        unique_classes, class_counts = np.unique(y_test, axis=0, return_counts=True)
        nr_classes = len(unique_classes)

        probs = clf.predict(X_test).detach().cpu().numpy().reshape(-1)
        test_predicitions = (probs > 0.5).astype(int)

        # Metriken
        test_auroc = roc_auc_score(y_test.tolist(), probs.tolist(), multi_class='ovo')
        test_accuracy = accuracy_score(y_test, test_predicitions)
        test_bal_acc = balanced_accuracy_score(y_test, test_predicitions)
        test_precision = precision_score(y_test, test_predicitions, zero_division=0)
        test_recall = recall_score(y_test, test_predicitions, zero_division=0)
        test_f1 = f1_score(y_test, test_predicitions, zero_division=0)  

        output_info_retest = {
            "cluster_id": cluster_id,
            "test_auroc": float(test_auroc),
            "test_accuracy": float(test_accuracy),
            "test_balanced_accuracy": float(test_bal_acc),
            "test_f1": float(test_f1),
            "test_recall": float(test_recall),
            "test_precision": float(test_precision),
        }

        Summary_retest.append(output_info_retest)
        
        pd.DataFrame({"y_true": y_test, "y_prob": probs, "y_pred": test_predicitions}).to_csv(
        os.path.join(cluster_path, f"predictions_{cluster_id}.csv"),
        index=False,
        )

    with open(os.path.join(first_check, f"summary_retest.json"), "w") as f:
        json.dump(Summary_retest, f, indent=2)

    print(f"[OK] Retest gespeichert in: {cluster_path}")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Ordner mit model.pt")
    p.add_argument("--target_column", required=True, help="Name der Target-spalte in test_data.csv")
    
    p.add_argument("--mode", required=False, default="classification")
    p.add_argument("--nr_blocks", required=False, default=2)
    p.add_argument("--hidden_size", required=False, default=128)
    p.add_argument("--dropout_rate", required=False, default=0.25)
    p.add_argument("--interpretable", required=False, action="store_true")
    args = p.parse_args()
    
    retest(args)