import gower
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json

from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import squareform

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def gower_hierarchical_clustering(X, y, categorical_cols, numerical_cols, plot_dendrogram=False, initial_cluster_len=None, clusters=None, one_more_division =True, offset=0):

    if initial_cluster_len is None:
        initial_cluster_len = len(X)


    if clusters is None:
        clusters = {}


    X_gower = X.copy()
    for col in numerical_cols:
        X_gower[col] = X_gower[col].astype(float)

    D = gower.gower_matrix(X_gower)
    D_condensed = squareform(D, checks=False)
    Z = linkage(D_condensed, method="average")

    if plot_dendrogram:
        plt.figure(figsize=(12, 5))
        dendrogram(Z, truncate_mode="level", p=5)
        plt.show()

    # Use silhouette score to find the best number of clusters
    print(f"Initial cluster size: {len(X_gower)}")
    best_k, _ = silhouette_score_criterion(D)

    # Initialize clusters

    if best_k != -1 and best_k > 1:

        cluster_labels = fcluster(Z, t=best_k, criterion='maxclust')

        print(f"Cluster divided into: {best_k}", "clusters")
        
        sub_cluster_len = len(X)

        cluster_sizes = {
            i: len(X[cluster_labels == i])
            for i in range(1, best_k + 1)
        }
        print(f"from {sub_cluster_len}, divided into:{cluster_sizes}")

        cluster_ratios = {
            i: size / initial_cluster_len
            for i, size in cluster_sizes.items()
        }

        big_clusters = [i for i, r in cluster_ratios.items() if r > 0.5]
        small_clusters = [i for i, r in cluster_ratios.items() if r < 0.10]
        normal_clusters = [
            i for i in cluster_ratios
            if i not in big_clusters and i not in small_clusters
        ]

        print(f"considered as big: {big_clusters}")
        print(f"considered as normal: {normal_clusters}")
        print(f"considered as small: {small_clusters}")

        from sklearn.preprocessing import LabelEncoder
        centroids = {}

        encoder = LabelEncoder()
        X_encoded = X.apply(encoder.fit_transform)

        for i in normal_clusters + small_clusters:
            c_small = X_encoded[cluster_labels == i].mean(axis=0)
            centroids[i] = c_small

        for i in normal_clusters:
            X_sub = X[cluster_labels == i]
            y_sub = y[cluster_labels == i]

            clusters[i+offset] = [X_sub, y_sub]
            print(f"cluster inserted {len(X_sub)}")
            show_clusters(clusters)
        
        
        for i in small_clusters:
            c_small = centroids[i]  

            if normal_clusters:
                print("estoy intentando juntar el puto:",small_clusters)
                # try:
                closest_normal = min(
                    normal_clusters,
                    key=lambda j: np.linalg.norm(c_small - centroids[j])
                )

                mask = cluster_labels == i

                X_small = X.loc[mask]  
                y_small = y.loc[mask]  

                clusters[closest_normal+offset][0] = pd.concat(
                    [clusters[closest_normal+offset][0], X_small],
                    axis=0
                )

                clusters[closest_normal+offset][1] = pd.concat(
                    [clusters[closest_normal+offset][1], y_small],
                    axis=0
                )
                print(f"cluster inserted {len(X_small)}")
                show_clusters(clusters)

            else:

                X_sub = X[cluster_labels == i]
                y_sub = y[cluster_labels == i]
                clusters[i+offset] = [X_sub, y_sub]

                print(f"cluster inserted {len(X_sub)}")
                show_clusters(clusters)

        print("----------------------------")

        for i in big_clusters:
            X_big = X[cluster_labels == i]
            y_big = y[cluster_labels == i]

            next_offset = len(X_big)

            sub_clusters, _ = gower_hierarchical_clustering(
                X_big,
                y_big,
                categorical_cols,
                numerical_cols,
                plot_dendrogram=False,
                initial_cluster_len=initial_cluster_len,
                clusters=clusters, 
                offset= next_offset
            )

    else:
        clusters[1] = (X, y)

    return clusters, D



def silhouette_score_criterion(gower_matrix,show_results=False):
    '''
    silhouette_score_criterion uses silhouette score to determine the optimal number of clusters (k) for hierarchical clustering.
    
    param:
        gower_matrix: Gower distance matrix
    return: 
        print k as number of clusters and silhouette score for each k
    '''
   
    best_score= 0

    for k in range(2, 8):
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage="average"
        )
        labels = model.fit_predict(gower_matrix)
        score = silhouette_score(gower_matrix, labels, metric="precomputed")
        if score > best_score:
            best_k = k
            best_score = score

        if show_results:
            print(f"silhouette analysis: k={k}, score={score}")

    if best_k >= 0.3:
        return best_k, best_score
    return -1, -1



def pca_mixed_data_visualization(clusters, categorical_cols, numerical_cols, visualization=False, dim=3):

    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    if not visualization:
        return

    #Whole graphic (All clusters)

    X_all = []
    cluster_ids = []

    for i, (X_cluster, _) in clusters.items():
        X_all.append(X_cluster)
        cluster_ids.extend([i] * len(X_cluster))

    X_all = pd.concat(X_all, axis=0)

    cluster_ids = np.array(cluster_ids)

    X_all_proc = preprocess.fit_transform(X_all)

    pca_global = PCA(n_components=dim)
    X_all_pca = pca_global.fit_transform(X_all_proc)

    if dim==2:
        plt.figure(figsize=(8, 6))
        for i in clusters.keys():
            mask = cluster_ids == i
            plt.scatter(
                X_all_pca[mask, 0],
                X_all_pca[mask, 1],
                alpha=0.7,
                label=f"Cluster {i}"
            )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        
        plt.title("PCA mixed data visualization - Global")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if dim==3:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        for i in clusters.keys():
            mask = cluster_ids == i
            ax.scatter(
                X_all_pca[mask, 0],
                X_all_pca[mask, 1],
                X_all_pca[mask, 2],
                alpha=0.7,
                label=f"Cluster {i}"
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.set_title("PCA mixed data visualization 3D - Global")
        ax.legend()

        plt.tight_layout()
        plt.show()



    #Per cluster graphic

    n_clusters = len(clusters)
    n_cols = 2
    n_rows = int(np.ceil(n_clusters / n_cols))

    if dim==2:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        axes = np.array(axes).reshape(-1)

        for idx, (i, (X_cluster, y_cluster)) in enumerate(clusters.items()):

            df = X_cluster.copy()
            X_proc = preprocess.fit_transform(df)

            pca = PCA(n_components=dim)
            X_pca = pca.fit_transform(X_proc)

            ax = axes[idx]
            ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA mixed data visualization - Cluster {i}")

            ax.text(
                0.02, 0.98,
                f"n = {len(df)}",
                transform=ax.transAxes,
                verticalalignment="top"
            )

    if dim==3:
        
        fig = plt.figure(figsize=(7 * n_cols, 5 * n_rows))
        axes = []

        for i in range(n_rows * n_cols):
            axes.append(fig.add_subplot(n_rows, n_cols, i + 1, projection="3d"))

        for idx, (i, (X_cluster, y_cluster)) in enumerate(clusters.items()):

            df = X_cluster.copy()
            X_proc = preprocess.fit_transform(df)

            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(X_proc)

            ax = axes[idx]
            ax.scatter(
                X_pca[:, 0],
                X_pca[:, 1],
                X_pca[:, 2],
                alpha=0.7
            )

            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_zlabel("PC3")
            ax.set_title(f"PCA mixed data visualization 3D - Cluster {i}")

            ax.text2D(
                0.02, 0.95,
                f"n = {len(df)}",
                transform=ax.transAxes
            )


    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()




def update_summary(path, new_entry):
    if os.path.exists(path):
        with open(path, 'r') as f:
            summary = json.load(f)
    else:
        summary = []

    summary.append(new_entry)

    with open(path, 'w') as f:
        json.dump(summary, f, indent=4)


def global_json_calculation(path, feature_list):

    with open(path, "r") as f:
        results = json.load(f)

    total_samples = 0
    
    weighted_train_auroc = 0.0
    weighted_train_accuracy = 0.0
    weighted_train_bal_acc = 0.0
    weighted_test_auroc = 0.0
    weighted_test_accuracy = 0.0
    weighted_test_bal_acc = 0.0
    weighted_test_f1 = 0.0
    weighted_test_precision = 0.0
    weighted_test_recall = 0.0
    weighted_train_time = 0.0
    weighted_inference_time = 0.0


    feature_ranking = {}
    feature_ranking = {feature: 0 for feature in feature_list}


    for data in results:

        if data.get("dataset_name") == "Cluster_whole":
            continue

        n = data["cluster_len"]

        weighted_train_auroc += data["train_auroc"] * n
        weighted_train_accuracy += data["train_accuracy"] * n
        weighted_train_bal_acc += data["train_balance_accuracy"] * n


        weighted_test_auroc += data["test_auroc"] * n
        weighted_test_accuracy += data["test_accuracy"] * n
        weighted_test_bal_acc += data["test_balance_accuracy"] * n
        weighted_test_f1 += data["test_f1"] * n
        weighted_test_precision += data["test_precision"] * n
        weighted_test_recall += data["test_recall"] * n

        weighted_train_time += data["train_time"]
        weighted_inference_time += data["inference_time"]

        total_samples += n

        top_features = data["top_features"]
        top_features_weights = data["top_features_weights"]

        for feature, weight in zip(top_features, top_features_weights):
            feature_ranking[feature] += weight * n


    weighted_train_auroc /= total_samples
    weighted_train_accuracy /= total_samples
    weighted_train_bal_acc /= total_samples

    weighted_test_auroc /= total_samples
    weighted_test_accuracy /= total_samples
    weighted_test_bal_acc /= total_samples
    weighted_test_f1 /= total_samples
    weighted_test_precision /= total_samples
    weighted_test_recall /= total_samples

    for feature in feature_ranking:
        feature_ranking[feature] /= total_samples

    feature_ranking = dict(sorted(feature_ranking.items(), key=lambda  x: x[1], reverse=True))

    global_metrics = {
        "train_auroc" : weighted_train_auroc,
        "train_accuracy": weighted_train_accuracy,
        "train_balance_accuracy": weighted_train_bal_acc,
        "test_auroc": weighted_test_auroc,
        "test_accuracy": weighted_test_accuracy,
        "test_balanced_accuracy": weighted_test_bal_acc,
        "test_f1": weighted_test_f1,
        "test_precision": weighted_test_precision,
        "test_recall": weighted_test_recall,
        "train_time": weighted_train_time,
        "inference_time": weighted_inference_time,
        "total_dataset_size": total_samples,
    }
    

    results.append({"cluster_global_mean": global_metrics})
    
    results.append({"cluster_global_ranking": feature_ranking})

    with open(path, "w") as f:
        json.dump(results, f, indent=2)



def show_clusters(clusters):
    for i, (X_data, y_data) in clusters.items():
        cluster_size = len(X_data)  # El tamaño del cluster es el número de filas en X_data
        print(f"Cluster size {i}: {cluster_size}")



def generate_cluster_feature_plots(summary_path):
    parent_dir = os.path.dirname(summary_path)

    with open(summary_path, "r") as f:
        summary_data = json.load(f)

    def plot_barh(features, weights, title, save_path):
        plt.figure(figsize=(8, 6))
        plt.barh(features, weights)
        plt.axvline(0)
        plt.xlabel("Feature weight")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # plots por cluster
    for item in summary_data:
        if "dataset_name" not in item:
            continue

        name = item["dataset_name"]
        
        if not name.lower().startswith("cluster"):
            continue

        features = item["top_features"]
        weights = item["top_features_weights"]

        features, weights = zip(
            *sorted(zip(features, weights), key=lambda x: x[1])
        )

        folder_name = name.lower().replace("cluster", "cluster")        
        cluster_dir = os.path.join(parent_dir, folder_name)

        os.makedirs(cluster_dir, exist_ok=True)

        save_path = os.path.join(cluster_dir, "top_features.png")

        plot_barh(
            features,
            weights,
            title=f"{name} Top Features",
            save_path=save_path
        )

    # plot global
    ranking = None
    for item in summary_data:
        if "cluster_global_ranking" in item:
            ranking = item["cluster_global_ranking"]
            break

    if ranking is None:
        return

    features = list(ranking.keys())
    weights = list(ranking.values())

    features, weights = zip(
        *sorted(zip(features, weights), key=lambda x: x[1])
    )

    summary_dir = os.path.join(parent_dir)
    os.makedirs(summary_dir, exist_ok=True)

    save_path = os.path.join(summary_dir, "cluster_global_ranking_mean.png")

    plot_barh(
        features,
        weights,
        title="Global Feature Ranking",
        save_path=save_path
    )


def save_test_data(info_cluster, output_directory,attribute_names):


    for cluster_id, info in info_cluster.items(): 
           
        X_test_cluster = info['X_test']
        
        y_test_cluster = info['y_test']

        X_test_cluster.columns = attribute_names

        y_test_cluster = pd.Series(y_test_cluster, name="churn")

        X_test_cluster = X_test_cluster.reset_index(drop=True)
        y_test_cluster = y_test_cluster.reset_index(drop=True)

        test_data = pd.concat([X_test_cluster, y_test_cluster], axis=1)
        
        print(
            f"Cluster {cluster_id} | "
            f"indices duplicados X: {test_data.index.duplicated().any()}"
        )

        folder_name = f"cluster_{cluster_id}"
        cluster_folder = os.path.join(output_directory, folder_name)
        os.makedirs(cluster_folder, exist_ok=True)

        csv_path = os.path.join(
            cluster_folder,
            f"test_data_cluster_{cluster_id}.csv")

        test_data.to_csv(csv_path, index=False)
