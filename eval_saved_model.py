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


def load_model_meta(model_dir: str) -> dict:
    """
    Lädt Meta-Infos zum Modell
    """
    meta_path = os.path.join(model_dir, "model_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} nicht gefunden. Bitte model_meta.json im Modellordner speichern."
        )
    with open(meta_path, "r") as f:
        return json.load(f)


def build_eval_out_dir(model_dir: str, test_csv: str) -> str:
    """
    Erzeugt einen Ordner neben model.pt
    """
    test_cluster_dir = os.path.dirname(test_csv)          # .../cluster_1
    test_run_dir = os.path.dirname(test_cluster_dir)      # .../0
    run_id = os.path.basename(test_run_dir)               # "0"
    cluster_name = os.path.basename(test_cluster_dir)     # "cluster_1"

    out_dir = os.path.join(model_dir, f"retest_on_run_{run_id}_{cluster_name}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def retest(model_dir: str, test_csv: str, target_column: str):

    # Load model + meta
    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    meta = load_model_meta(model_dir)

    # Load test data
    df = pd.read_csv(test_csv)
    if target_column not in df.columns:
        raise KeyError(f"Target '{target_column}' nicht in {test_csv} gefunden.")

    y_test = df[target_column].to_numpy().astype(int)
    X_test = df.drop(columns=[target_column]).to_numpy().astype(np.float32)

    # Dummy args für Classifier
    args = argparse.Namespace(
        mode=meta.get("mode", "classification"),
        nr_epochs=0,
        batch_size=0,
        learning_rate=0.0,
        weight_decay=0.0,
        scheduler_t_mult=2,
        nr_restarts=1,
        weight_norm=0.0,
        augmentation_probability=0.0,
        dropout_rate=float(meta.get("dropout_rate", 0.25)),
        disable_wandb=True,
    )

    interpretable = bool(meta.get("interpretable", True))
    model_name = "inn" if interpretable else "tabresnet"

    # Netzwerk muss gleich aufgebaut werden wie beim Training
    network_configuration = {
        "nr_features": int(X_test.shape[1]),
        "nr_classes": 1,  # bei Telco => 1
        "nr_blocks": int(meta["nr_blocks"]),
        "hidden_size": int(meta["hidden_size"]),
        "dropout_rate": float(meta.get("dropout_rate", 0.25)),
    }

    # Für predict reicht das als Platzhalter
    categorical_indicator = [False] * X_test.shape[1]
    attribute_names = [f"f{i}" for i in range(X_test.shape[1])]

    clf = Classifier(
        network_configuration=network_configuration,
        args=args,
        categorical_indicator=categorical_indicator,
        attribute_names=attribute_names,
        model_name=model_name,
        output_directory=model_dir,
        disable_wandb=True,
    )

    # weights laden
    state = torch.load(model_path)
    clf.model.load_state_dict(state)
    clf.model.eval()

    clf.ensemble_snapshots = [state]

    probs = clf.predict(X_test).detach().cpu().numpy().reshape(-1)

    test_auroc = roc_auc_score(y_test.tolist(), probs.tolist(), multi_class="ovo")
    test_pred = (probs > 0.5).astype(int)

    # Metriken
    # ToDO: Teil in main_experiment auslagern, damit nicht doppelt
    test_accuracy = accuracy_score(y_test, test_pred)
    test_bal_acc = balanced_accuracy_score(y_test, test_pred)
    test_precision = precision_score(y_test, test_pred, zero_division=0)
    test_recall = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)

    output_info = {
        "test_auroc": float(test_auroc),
        "test_accuracy": float(test_accuracy),
        "test_balanced_accuracy": float(test_bal_acc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
    }

    out_dir = build_eval_out_dir(model_dir, test_csv)

    # Speichere Referenz
    with open(os.path.join(out_dir, "retest_meta.json"), "w") as f:
        json.dump(
            {
                "model_dir": model_dir,
                "model_path": model_path,
                "test_csv": test_csv,
                "target_column": target_column,
                "n_test": int(len(y_test)),
            },
            f,
            indent=2,
        )

    # Speichere output_info
    with open(os.path.join(out_dir, "output_info_retest.json"), "w") as f:
        json.dump(output_info, f, indent=2)

    # Predictions speichern (vielleicht für spätere Vergleiche)
    pd.DataFrame({"y_true": y_test, "y_prob": probs, "y_pred": test_pred}).to_csv(
        os.path.join(out_dir, "predictions.csv"),
        index=False,
    )

    print(f"[OK] Retest gespeichert in: {out_dir}")
    print(json.dumps(output_info, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Ordner mit model.pt und model_meta.json")
    p.add_argument("--test_csv", required=True, help="Pfad zu test_data.csv (preprocessd)")
    p.add_argument("--target_column", required=True, help="Name der Target-Spalte in test_data.csv")
    args = p.parse_args()

    retest(args.model_dir, args.test_csv, args.target_column)