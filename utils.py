import os
from typing import Dict, List, Tuple

import numpy as np
import openml
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OrdinalEncoder,
    LabelEncoder,
    StandardScaler,
    OneHotEncoder,
    TargetEncoder,
)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def prepare_data_for_cutmix(
    x: torch.Tensor,
    y: torch.Tensor,
    augmentation_prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply the cutmix augmentation to the data.

    Args:
        x: The examples.
        y: The labels.
        augmentation_prob: The probability with which to apply the operation.

    Returns:
        x: The augmented examples.
        y: The labels.
        y_shuffled: The shuffled labels.
        lam: The lambda value for the augmentation operation.
    """
    # Shuffle the data
    indices = torch.randperm(x.size(0))
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # Generate the lambda value
    lam = torch.distributions.beta.Beta(1, 1).sample()

    if np.random.rand() > augmentation_prob:
        lam = 1
    else:
        # Generate the mixup mask per example and feature
        for i in range(x.size(0)):
            cut_column_indices = torch.as_tensor(
                np.random.choice(
                    range(x.size(1)),
                    max(1, np.int32(x.size(1) * (1 - lam))),
                    replace=False,
                ),
                dtype=torch.int64,
            )

            x[i, cut_column_indices] = x_shuffled[i, cut_column_indices]

    return x, y, y_shuffled, lam


def prepare_data_for_mixup(
    x: torch.Tensor,
    y: torch.Tensor,
    numerical_features: List,
    augmentation_prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply the mixup augmentation to the data.

    Args:
        x: The examples.
        y: The labels.
        numerical_features: The numerical features.
        augmentation_prob: The probability with which to apply the operation.

    Returns:
        x: The augmented examples.
        y: The labels.
        y_shuffled: The shuffled labels.
        lam: The lambda value for the augmentation operation.
    """
    # Shuffle the data
    indices = torch.randperm(x.size(0))
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    # Generate the lambda value
    lam = torch.distributions.beta.Beta(1, 1).sample()

    if np.random.rand() > augmentation_prob:
        lam = 1
    else:
        # Generate the mixup mask per example and numerical feature
        for i in range(x.size(0)):
            cut_column_indices = torch.as_tensor(
                np.random.choice(
                    numerical_features,
                    max(1, np.int32(len(numerical_features) * (1 - lam))),
                    replace=False,
                ),
                dtype=torch.int64,
            )
            x[i, cut_column_indices] = lam * x[i, cut_column_indices] + (1. - lam) * x_shuffled[i, cut_column_indices]

    return x, y, y_shuffled, lam


def prepare_data_for_cutout(
    x: torch.Tensor,
    y: torch.Tensor,
    numerical_features: List,
    augmentation_prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply the cutout augmentation to the data.

    Args:
        x: The examples.
        y: The labels.
        numerical_features: The numerical features.
        augmentation_prob: The probability with which to apply the operation.

    Returns:
        x: The augmented examples.
        y: The labels.
        y_shuffled: The shuffled labels.
        lam: The lambda value for the augmentation operation.
    """
    # Shuffle the data
    indices = torch.randperm(x.size(0))
    y_shuffled = y[indices]

    # Generate the lambda value
    lam = torch.distributions.beta.Beta(1, 1).sample()

    if np.random.rand() > augmentation_prob:
        lam = 1
    else:
        # Generate the mixup mask per example and feature
        for i in range(x.size(0)):
            cut_column_indices = np.random.choice(
                range(x.size(1)),
                max(1, np.int32(x.size(1) * lam)),
                replace=False,
            )

            cut_cat_indices = [i for i in cut_column_indices if i not in numerical_features]
            cut_numerical_indices = [i for i in cut_column_indices if i in numerical_features]

            if len(cut_cat_indices) > 0:

                cut_cat_indices = torch.as_tensor(
                    cut_cat_indices,
                    dtype=torch.int64,
                )
                x[i, cut_cat_indices] = 0

            if len(cut_numerical_indices) > 0:
                cut_numerical_indices = torch.as_tensor(
                    cut_numerical_indices,
                    dtype=torch.int64,
                )

                x[i, cut_numerical_indices] = 0

    return x, y, y_shuffled, lam


def fgsm_attack(
    x: torch.Tensor,
    y: torch.Tensor,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    augmentation_prob: float,
    epsilon: float,
) -> torch.Tensor:
    """Generate adversarial examples using the FGSM attack.

    Args:
        x: The examples.
        y: The labels.
        model: The trained model.
        criterion: The criterion with which the model was trained.
        augmentation_prob: The probability with which to apply the operation.
        epsilon: The epsilon value for the FGSM attack.

    Returns:
        x: The augmented examples.
    """
    if np.random.rand() > augmentation_prob:
        return x
    else:
        # copy tensor to avoid changing the original one
        x = x.clone().detach().requires_grad_(True)

        # perform the attack
        outputs = model(x)
        if outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
        cost = criterion(outputs, y)

        grad = torch.autograd.grad(cost, x, retain_graph=False, create_graph=False)[0]

        adv_data = x + epsilon * grad.sign()

    return adv_data


def random_noise(
    x: torch.Tensor,
    y: torch.Tensor,
    augmentation_prob: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply random noise to the data.

    Args:
        x: The examples.
        y: The labels.
        augmentation_prob: The probability with which to apply the operation.

    Returns:
        x: The augmented examples.
        y: The labels.
        y: The labels.
        lam: The lambda value for the augmentation operation.
    """
    # Generate the lambda value
    lam = torch.distributions.beta.Beta(1, 1).sample()

    if np.random.rand() > augmentation_prob:
        pass
    else:
        # Generate the mixup mask per example and feature
        for i in range(x.size(0)):
            cut_column_indices = torch.as_tensor(
                np.random.choice(
                    range(x.size(1)),
                    max(1, np.int32(x.size(1) * (1 - lam))),
                    replace=False,
                ),
                dtype=torch.int64,
            )
            x[i, cut_column_indices] = torch.add(
                x[i, cut_column_indices],
                (0.1 ** 0.5) * torch.randn(x[i, cut_column_indices].shape).to(x.device),
            )

    return x, y, y, 1


def augment_data(
    x: torch.Tensor,
    y: torch.Tensor,
    numerical_features: List,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    augmentation_prob: float = 0.5,
) -> Tuple:
    """Perform data augmentation.

    Args:
        x: The examples.
        y: The labels.
        numerical_features: The numerical features.
        model: The trained model.
        criterion: The criterion with which the model was trained.
        augmentation_prob: The probability with which to apply the operation.

    Returns:
        x: The augmented examples.
        y: The labels.
        y_shuffled: The shuffled labels.
        lam: The lambda value for the augmentation operation.
    """
    augmentation_types = {
        1: "mixup",
        2: "cutout",
        3: "cutmix",
        4: "fgsm",
        5: "random_noise",
    }

    if len(numerical_features) == 0:
        """remove mixup from the list of augmentation types
        since it makes more sense for numerical features"""
        del augmentation_types[1]

    augmentation_type = augmentation_types[np.random.randint(1, len(augmentation_types) + 1)]
    if augmentation_type == "cutmix":
        return prepare_data_for_cutmix(x, y, augmentation_prob)
    elif augmentation_type == "mixup":
        return prepare_data_for_mixup(x, y, numerical_features, augmentation_prob)
    elif augmentation_type == "cutout":
        return prepare_data_for_cutout(x, y, numerical_features, augmentation_prob)
    elif augmentation_type == "fgsm":
        return x, fgsm_attack(x, y, model, criterion, augmentation_prob, 0.007), y, y, 0.5
    elif augmentation_type == "random_noise":
        return random_noise(x, y, augmentation_prob)
    else:
        raise ValueError("The augmentation type must be one of 'mixup', "
                         "'cutout', 'cutmix', 'fgsm', or 'random_noise'")


def preprocess_dataset(
    X: pd.DataFrame,
    y: pd.DataFrame,
    encode_categorical: bool,
    categorical_indicator: List,
    attribute_names: List,
    test_split_size: float = 0.2,
    seed: int = 11,
    encoding_type: str = "ordinal",
    hpo_tuning: bool = False,
) -> Dict:
    """Preprocess the dataset.

    Args:
        X: The examples.
        y: The labels.
        encode_categorical: Whether to encode the categorical features.
        categorical_indicator: An indicator for differentiating between categorical
            and numerical features.
        attribute_names: The names of the features.
        test_split_size: The size of the test split.
        seed: The random seed.
        encoding_type: The encoding type for the categorical features. Whether it should be
            'ordinal' or 'one-hot'.
        hpo_tuning: Whether to create a validation set for hyperparameter optimization.

    Returns:
        info_dict: A dictionary with the preprocessed data and additional information
    """
    dropped_column_names = []
    dropped_column_indices = []

    for column_index, column_name in enumerate(X.keys()):
        # if more than 90% of the values are missing, mark the column
        if X[column_name].isnull().sum() > len(X[column_name]) * 0.9:
            dropped_column_names.append(column_name)
            dropped_column_indices.append(column_index)
        # if the column has only one unique value, mark the column
        if X[column_name].nunique() == 1:
            dropped_column_names.append(column_name)
            dropped_column_indices.append(column_index)

    for column_index, column_name in enumerate(X.keys()):
        if X[column_name].dtype == 'object' or X[column_name].dtype == 'category' or X[column_name].dtype == 'string':
            # if more than 90% of the values are unique, mark the column
            if X[column_name].nunique() / len(X[column_name]) > 0.9:
                dropped_column_names.append(column_name)
                dropped_column_indices.append(column_index)

    # drop the marked columns
    X = X.drop(dropped_column_names, axis=1)

    # account for dropped columns and match the different indicators
    attribute_names = [attribute_name for attribute_name in attribute_names if attribute_name not in dropped_column_names]
    categorical_indicator = [categorical_indicator[i] for i in range(len(categorical_indicator)) if i not in dropped_column_indices]

    column_category_values = []

    # take pandas categories into account
    for cat_indicator, column_name in zip(categorical_indicator, X.keys()):
        if cat_indicator:
            column_categories = list(X[column_name].cat.categories)
            column_category_values.append(column_categories)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_split_size,
        random_state=seed,
        stratify=y,
    )

    if hpo_tuning:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=test_split_size / (1 - test_split_size),
            random_state=seed,
            stratify=y_train,
        )

    # pandas series number of unique values
    nr_classes = y_train.nunique()

    # scikit learn label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    if hpo_tuning:
        y_valid = label_encoder.transform(y_valid)

    numerical_features = [i for i in range(len(categorical_indicator)) if not categorical_indicator[i]]
    categorical_features = [i for i in range(len(categorical_indicator)) if categorical_indicator[i]]

    # save the column types
    column_types = {}
    for column_name in X_train.keys():
        if X_train[column_name].dtype == 'object' or X_train[column_name].dtype == 'category' or X_train[column_name].dtype == 'string':
            column_types[column_name] = 'category'
        elif pd.api.types.is_numeric_dtype(X_train[column_name]):
            column_types[column_name] = 'float64'
        else:
            raise ValueError("The column type must be one of 'object', 'category', 'string', 'int' or 'float'")

    dataset_preprocessors = []
    if len(numerical_features) > 0:
        numerical_preprocessor = ('numerical', StandardScaler(), numerical_features)
        dataset_preprocessors.append(numerical_preprocessor)

    if len(categorical_features) > 0 and encode_categorical:
        if nr_classes > 2:
            if encoding_type == "ordinal":
                categorical_preprocessor = (
                    'categorical_encoder',
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        categories=column_category_values,
                    ),
                    categorical_features,
                )
            else:
                categorical_preprocessor = (
                    'categorical_encoder',
                    OneHotEncoder(
                        handle_unknown='ignore',
                        sparse=False,
                        categories=column_category_values,
                        drop='if_binary',
                    ),
                    categorical_features,
                )
        else:
            categorical_preprocessor = (
                'categorical_encoder',
                TargetEncoder(random_state=seed),
                categorical_features,
            )
        dataset_preprocessors.append(categorical_preprocessor)

    column_transformer = ColumnTransformer(
        dataset_preprocessors,
        remainder='passthrough',
    )
    X_train = column_transformer.fit_transform(X_train, y_train)
    X_test = column_transformer.transform(X_test)

    if hpo_tuning:
        X_valid = column_transformer.transform(X_valid)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    if hpo_tuning:
        X_valid = pd.DataFrame(X_valid)

    if len(numerical_features) > 0:
        new_categorical_indicator = [False] * len(numerical_features)
        new_attribute_names = [attribute_names[i] for i in numerical_features]
    else:
        new_categorical_indicator = []
        new_attribute_names = []

    if len(categorical_features) > 0:
        if nr_classes == 2:
            new_categorical_indicator.extend([True] * len(categorical_features))
            new_attribute_names.extend([attribute_names[i] for i in categorical_features])
        else:
            for i in range(len(column_category_values)):
                nr_unique_categories = len(column_category_values[i])
                if nr_unique_categories > 2:
                    new_categorical_indicator.extend([True] * len(column_category_values[i]))
                    new_attribute_names.extend([attribute_names[categorical_features[i]] + '_' + str(category) for category in column_category_values[i]])
                else:
                    new_categorical_indicator.extend([True])
                    new_attribute_names.extend([attribute_names[categorical_features[i]]])

    if encode_categorical:
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
    else:
        for cat_indicator, column_name in zip(categorical_indicator, X_train.keys()):
            if not cat_indicator:
                X_train[column_name] = X_train[column_name].fillna(0)
                X_test[column_name] = X_test[column_name].fillna(0)
                if hpo_tuning:
                    X_valid[column_name] = X_valid[column_name].fillna(0)
            else:
                X_train[column_name] = X_train[column_name].cat.add_categories('missing')
                X_train[column_name].cat.reorder_categories(np.roll(X_train[column_name].cat.categories, 1))
                X_train[column_name] = X_train[column_name].fillna('missing')

                X_test[column_name] = X_test[column_name].cat.add_categories('missing')
                X_test[column_name].cat.reorder_categories(np.roll(X_test[column_name].cat.categories, 1))
                X_test[column_name] = X_test[column_name].fillna('missing')

                if hpo_tuning:
                    X_valid[column_name] = X_valid[column_name].cat.add_categories('missing')
                    X_valid[column_name].cat.reorder_categories(np.roll(X_valid[column_name].cat.categories, 1))
                    X_valid[column_name] = X_valid[column_name].fillna('missing')

    info_dict = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'categorical_indicator': new_categorical_indicator,
        'attribute_names': new_attribute_names,
    }

    if hpo_tuning:
        info_dict['X_valid'] = X_valid
        info_dict['y_valid'] = y_valid

    return info_dict


def get_dataset(
    dataset_id: int,
    test_split_size: float = 0.2,
    seed: int = 11,
    encode_categorical: bool = True,
    encoding_type: str = 'ordinal',
    hpo_tuning: bool = False,
) -> Dict:
    """Get/Preprocess the dataset.

    Args:
        dataset_id: The dataset identifier.
        test_split_size: The size of the test split.
        seed: The random seed.
        encode_categorical: Whether to encode the categorical features.
        encoding_type: The encoding type for the categorical features. Whether it should be
            'ordinal' or 'one-hot'.
        hpo_tuning: Whether to create a validation set for hyperparameter optimization.

    Returns:
        info_dict: A dictionary with the preprocessed data and additional information
    """
    # Get the data
    dataset = openml.datasets.get_dataset(dataset_id, download_data=False)              #TODO
    dataset_name = dataset.name
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='dataframe',
        target=dataset.default_target_attribute,
    )
    info_dict = preprocess_dataset(
        X,
        y,
        encode_categorical,
        categorical_indicator,
        attribute_names,
        test_split_size=test_split_size,
        seed=seed,
        encoding_type=encoding_type,
        hpo_tuning=hpo_tuning,
    )
    info_dict['dataset_name'] = dataset_name

    return info_dict

# Neue Hilfsfunktion zum Einlesen einer lokalen CSV
def get_dataset_from_csv(
    csv_path: str,
    target_column: str,
    test_split_size: float = 0.2,
    seed: int = 11,
    encode_categorical: bool = True,
    encoding_type: str = 'ordinal',
    hpo_tuning: bool = False,
    n_clusters: int = 1, # für Clustering
) -> Dict:
    """Load and preprocess a dataset from a local CSV file."""

    df = pd.read_csv(csv_path)

    # Zielvariable
    y = df[target_column]
    y = y.str.strip() if y.dtype == 'object' else y

    # Feature-Matrix
    X = df.drop(columns=[target_column]).copy()
    if 'customerID' in X.columns:
        X = X.drop(columns=['customerID'])


    #Fix für Error
    #ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 128])
    min_samples_per_class = 2

    vc = y.value_counts()
    valid_classes = vc[vc >= min_samples_per_class].index

    mask = y.isin(valid_classes)
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    categorical_indicator = X.dtypes == 'object'
    attribute_names = X.columns.to_numpy()

    # Features in 'category' konvertieren
    for col, is_cat in zip(X.columns, categorical_indicator):
        if is_cat:
            X[col] = X[col].astype('category')

    # Wie get_data Funktion, Preprocessing
    # Clustering
    if n_clusters > 1:
        clusters, cluster_labels = split_dataset_by_clustering(
            X,
            y,
            n_clusters=n_clusters,
            random_state=seed,
        )

        cluster_infos = {}
        for cid, (X_c, y_c) in clusters.items():
            info = preprocess_dataset(
                X_c,
                y_c,
                encode_categorical,
                categorical_indicator.to_numpy(),
                attribute_names,
                test_split_size=test_split_size,
                seed=seed,
                encoding_type=encoding_type,
                hpo_tuning=hpo_tuning,
            )
            info["dataset_name"] = f"{os.path.basename(csv_path)}_cluster_{cid}"
            cluster_infos[cid] = info

        return {
            "clustered": True,
            "clusters": cluster_infos,
            "n_clusters": n_clusters,
            "cluster_labels": cluster_labels,
            "X_for_clustering": X.select_dtypes(include=[np.number]),
        }

    # kein Clustering
    info = preprocess_dataset(
        X,
        y,
        encode_categorical,
        categorical_indicator.to_numpy(),
        attribute_names,
        test_split_size=test_split_size,
        seed=seed,
        encoding_type=encoding_type,
        hpo_tuning=hpo_tuning,
    )

    info["dataset_name"] = os.path.basename(csv_path)
    return info

def split_dataset_by_clustering(
    X: pd.DataFrame,
    y: pd.Series,
    n_clusters: int = 3,
    random_state: int = 11,
):
    """
    Split dataset into K clusters using KMeans (feature-based, unsupervised).

    Returns:
        Dict[int, Tuple[X_cluster, y_cluster]]
    """

    # Nur numerische Features fürs Clustering
    X_num = X.select_dtypes(include=[np.number])

    if X_num.shape[1] == 0:
        raise ValueError("Clustering requires at least one numerical feature.")

    # Skalieren (sehr wichtig)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    # KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Aufteilen
    clustered_data = {}
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        clustered_data[cluster_id] = (
            X.loc[mask].reset_index(drop=True),
            y.loc[mask].reset_index(drop=True),
        )

    return clustered_data, cluster_labels

class BasicBlock(nn.Module):

    def __init__(
        self,
        in_features: int,
        output_features: int,
        dropout_rate: float,
    ):
        """A basic residual block.

        Args:
            in_features: Number of input features to the first
                layer of the residual block.
            output_features: Number of output features
            dropout_rate: Dropout rate for the residual block.
        """
        super(BasicBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.hidden_state_dropout = nn.Dropout(self.dropout_rate)
        self.residual_dropout = nn.Dropout(self.dropout_rate)
        self.linear1 = nn.Linear(in_features, output_features)
        self.bn1 = nn.BatchNorm1d(output_features)
        self.linear2 = nn.Linear(output_features, output_features)
        self.bn2 = nn.BatchNorm1d(output_features)
        self.gelu = nn.GELU()

    def forward(self, x) -> torch.Tensor:

        residual = x
        residual = self.residual_dropout(residual)

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.hidden_state_dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += residual
        out = self.gelu(out)

        return out


def make_residual_block(
    in_features: int,
    output_features: int,
    dropout_rate: float = 0.25,
) -> BasicBlock:
    """Creates a residual block.

    Args:
        in_features: Number of input features to the first
            layer of the residual block.
        output_features: Number of output features
            for the last layer of the residual block.
        dropout_rate: Dropout rate for the residual block.

    Returns:
        A residual block.
    """
    return BasicBlock(in_features, output_features, dropout_rate)

# Hilfsfunktion für die Ordnerstruktur
def get_next_run_id(base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    existing = [
        int(d) for d in os.listdir(base_dir)
        if d.isdigit() and os.path.isdir(os.path.join(base_dir, d))
    ]
    return str(max(existing) + 1) if existing else "0"

# Balkendiagramm für die Gewichte
def plot_top_features(
    output_info: dict,
    out_dir: str,
    top_k: int = 15,
    filename: str = "top_features.png",
):
    if "top_features" not in output_info:
        return  # nichts zu plotten (z.B. nicht-interpretable Modell)

    features = output_info["top_features"][:top_k]
    weights = output_info["top_features_weights"][:top_k]

    plt.figure(figsize=(8, 5))
    plt.barh(features[::-1], weights[::-1])  # umdrehen für schöne Sortierung
    plt.xlabel("Feature Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

def plot_clusters_pca(
    X_num: pd.DataFrame,
    cluster_labels: np.ndarray,
    out_dir: str,
    filename: str = "clusters_pca.png",
):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    pca = PCA(n_components=2, random_state=11)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 6))
    plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.7,
    )

    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("Cluster visualization (PCA)")
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

